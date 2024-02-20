import torch
import torch.nn as nn
import numpy as np
import os
import hdf5storage as hdf5


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)


# Validate
def validate(val_loader, model, mrae, rmse, psnr, sample_folder, is_save=False):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input_rgb, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input_rgb = input_rgb.cuda()
            target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input_rgb)
            loss_mrae = mrae(output, target)
            loss_rmse = rmse(output, target)
            loss_psnr = psnr(output, target)

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

        if is_save:
            # Save
            output = output.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
            save_img_name = str(i) + '.mat'
            save_img_path = os.path.join(sample_folder, save_img_name)
            hdf5.write(data=output, path='cube', filename=save_img_path, matlab_compatible=True)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg