from torchsummary import summary
import torch
import torch.nn as nn
from torch.fft import fft, ifft
import torch.nn.functional as F
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim*mult, 1, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim*mult, dim*mult, 3, 1, 1, groups=dim*mult, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim*mult, dim, 1, 1, bias=bias)
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        out = self.ff(x.permute(0, 3, 1, 2))
        return out


class Attention(nn.Module):
    def __init__(self, dim, win_size, depth, bias=False):
        super().__init__()
        self.window_size = [win_size, win_size]
        self.depth = depth
        self.shift_size = self.window_size[0] // 2
        self.rescale1 = nn.Parameter(torch.ones(1, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(1, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, 3, 1, 1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        
    def forward(self, x, q=None):
        """
        x: [b, c, h, w] | [b, 32, 128, 128]
        return: [b, c, h, w] | [b, 32, 128, 128]
        """
        if self.depth % 2:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        b, c, h, w = x.shape
        q = rearrange(q, 'b c h w -> b 1 c (h w)')
        x = rearrange(x, 'b c (h b0) (w b1) -> (b h w) c b0 b1', b0=self.window_size[0], b1=self.window_size[1])
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        k = rearrange(k, '(b h w) c h1 w1 -> b (h w) c (h1 w1)', h=h//self.window_size[0], w=w//self.window_size[1])
        v = rearrange(v, '(b h w) c h1 w1 -> b (h w) c (h1 w1)', h=h//self.window_size[0], w=w//self.window_size[1])
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn1 = (k @ q.transpose(-2, -1)) * self.rescale1
        attn1 = attn1.softmax(dim=-1)
        out1 = attn1 @ v

        attn2 = (k @ v.transpose(-2, -1)) * self.rescale2
        attn2 = attn2.softmax(dim=-1)
        out2 = attn2 @ q

        # TODO: 添加快速傅里叶变化, 虚部与实部直接相加处理
        out1 = fft(out1.to(torch.complex64))
        out2 = fft(out2.to(torch.complex64))
        out = ifft(out1 + out2)
        out = (out.real + out.imag).to(torch.float32)

        out = rearrange(out, 'b (h w) c (h1 w1) -> (b h w) c h1 w1', h=h//self.window_size[0], w=w//self.window_size[1], h1=self.window_size[0], w1=self.window_size[1])
        out = rearrange(out, '(b h w) c b0 b1 -> b c (h b0) (w b1)', h=h//self.window_size[0], w=w//self.window_size[1], b0=self.window_size[0], b1=self.window_size[1])
        out = self.project_out(out)

        if self.depth % 2:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return out


class GExtract(nn.Module):
    def __init__(self, conv_size, win_size, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(conv_size, conv_size*3, 1, 1, bias=bias)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(conv_size*3, conv_size*3, 3, 1, 1, bias=bias, groups=conv_size*3)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(conv_size*3, conv_size, 1, 1, bias=bias)
        self.maxPool = nn.MaxPool2d(win_size, win_size)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        x = self.maxPool(x)
        return x


class TRAN(nn.Module):
    def __init__(self, dim, in_size, bias=False):
        super().__init__()
        if in_size == 128:
            g_size = 8
        elif in_size == 64:
            g_size = 4
        else:
            g_size = 2
        w_size = 16

        self.blocks = nn.ModuleList([])
        for i in range(3):
            self.blocks.append(
                nn.ModuleList([
                    GExtract(dim, g_size),
                    Attention(dim, w_size, i + 1, bias=bias),
                    FeedForward(dim, bias=bias)
                ])
            )
            w_size = w_size // 2
            g_size = g_size * 2
    
    def forward(self, x):
        """
        x: [b, c, h, w] | [b, 32, 128, 128]
        return: [b, c, h, w] | [b, 32, 128, 128]
        """
        for (g, a, f) in self.blocks:
            fea_g = g(x)
            x = x + a(x, fea_g)
            x = x + f(x)
        return x
    

class MSTV3Block(nn.Module):
    def __init__(self, mid_channels=32, bias=False):
        super().__init__()
        # Input projection
        self.embedding = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=bias)
        # Output projection
        self.mapping = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=bias)
        # Encoder
        self.encode1 = TRAN(32, 128, bias=bias)
        self.down1 = nn.Conv2d(mid_channels, mid_channels*2, 4, 2, 1, bias=bias)
        self.encode2 = TRAN(64, 64, bias=bias)
        self.down2 = nn.Conv2d(mid_channels*2, mid_channels*4, 4, 2, 1, bias=bias)
        # Middle layer
        self.encode3 = TRAN(128, 32, bias=bias)
        # Decoder
        self.convT1 = nn.ConvTranspose2d(mid_channels*4, mid_channels*2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.up1 = nn.Conv2d(mid_channels*4, mid_channels*2, 1, 1, bias=bias)
        self.decode1 = TRAN(64, 64)
        self.convT2 = nn.ConvTranspose2d(mid_channels*2, mid_channels, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.up2 = nn.Conv2d(mid_channels*2, mid_channels, 1, 1, bias=bias)
        self.decode2 = TRAN(32, 128)
        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
    def forward(self, x):
        """
        x: [b, c, h, w] | [b, 32, 128, 128]
        return: [b, c, h, w] | [b, 32, 128, 128]
        """
        # Input projection
        fea = self.embedding(x)
        # Encode process
        fea_encoder = []
        fea = self.encode1(fea)
        fea_encoder.append(fea)
        fea = self.down1(fea)
        fea = self.encode2(fea)
        fea_encoder.append(fea)
        fea = self.down2(fea)
        fea = self.encode3(fea)
        # Decode process
        fea = self.convT1(fea)
        fea = self.up1(torch.cat([fea, fea_encoder[1]], dim=1))
        fea = self.decode1(fea)
        fea = self.convT2(fea)
        fea = self.up2(torch.cat([fea, fea_encoder[0]], dim=1))
        fea = self.decode2(fea)
        # Output projection
        out = self.mapping(fea) + x
        return out


class Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, mid_channels=32, stage=3, bias=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=bias)
        
        self.basebone = nn.ModuleList([])
        for i in range(stage):
            self.basebone.append(MSTV3Block())
        
        self.conv_out = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        """
        x: [b, c, h, w] | [b, 3, 128, 128]
        return: [b, c, h, w] | [b, 31, 128, 128]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        x = self.conv_in(x)
        for i in range(len(self.basebone)):
            x = self.basebone[i](x)
        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    model = Network()
    x = torch.rand(2, 3, 128, 128)
    model(x)
    # summary(model, x)
