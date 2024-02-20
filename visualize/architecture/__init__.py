import torch
from .awan import AWAN
from .gdlnet import Network
from .hrnet import SGN
from .hscnn_plus import HSCNN_Plus
from .mst_plus_plus import MST_Plus_Plus


def model_generator(method, pretrained_model_path):
    if method == "awan":
        model = AWAN()
    elif method == "gdlnet":
        model = Network()
    elif method == "hrnet":
        model = SGN()
    elif method == "hscnn_plus":
        model = HSCNN_Plus()
    elif method == "mst_plus_plus":
        model = MST_Plus_Plus()
    else:
        raise ValueError("Invalid method: {}".format(method))
    if pretrained_model_path is not None:
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    # model.load_state_dict(torch.load(pretrained_model_path))
    # model = model.cuda()
    # model.eval()
    return model
