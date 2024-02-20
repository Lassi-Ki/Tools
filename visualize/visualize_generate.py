import torch
import argparse
import os
import cv2
import hdf5storage as hdf5
import numpy as np
from architecture import model_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fcl")
    parser.add_argument("--pretrained_model_path", type=str,
                        default=r"D:\Codes\Tools\models\FCL_v3_0438.pth")
    parser.add_argument("gpu_id", type=str, default='0')
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--input_dir", type=str,
                        default=r"./datas")
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    output_dir = os.path.join(opt.output_dir, opt.model)
    os.makedirs(output_dir, exist_ok=True)

    model = model_generator(opt.model, opt.pretrained_model_path)
    print('Parameters number is: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    bgr = cv2.imread(opt.inputt_dir)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb)
    # rgb = rgb / 255.0
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = rgb[None, :]

    input_rgb = torch.from_numpy(rgb)

    if torch.cuda.is_available():
        input_rgb = input_rgb.cuda()

    output_spectral = model(input_rgb)
    output_spectral = output_spectral.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)

    save_mat_name = os.path.join(output_dir, 'output_spectral.mat')

    hdf5.write(data=output_spectral, path='cube', filename=save_mat_name, matlab_compatible=True)
