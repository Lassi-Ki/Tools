import argparse
import os
import datasets
from torch.utils.data import DataLoader
from visualize import utils
from architecture import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
    parser.add_argument('--method', type=str, default='fcl')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=r'D:\Codes\Tools\models\FCL_v3_0438.pth')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--dataset_mode", type=str, default="NTIRE2022")
    parser.add_argument("--dataset_path", type=str, default=r"")
    parser.add_argument("--patch_size", type=int, default=128)
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    output_dir = os.path.join(opt.output_dir, opt.method)
    os.makedirs(output_dir, exist_ok=True)

    model = model_generator(opt.method, opt.pretrained_model_path)
    print('Parameters number is: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    mrae = utils.Loss_MRAE()
    rmse = utils.Loss_RMSE()
    psnr = utils.Loss_PSNR()

    if torch.cuda.is_available():
        model = model.cuda()
        mrae = mrae.cuda()
        rmse = rmse.cuda()
        psnr = psnr.cuda()

    if opt.dataset_mode == "NTIRE2022":
        val_data = datasets.NTIRE2022Dataset(opt.dataset_path, opt.patch_size)
    elif opt.dataset_mode == "CAVE":
        pass
        # val_data = datasets.CAVE(opt.dataset_path, opt.patch_size)
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % opt.dataset_mode)
    print("Validation set samples: ", len(val_data))
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, pin_memory=False)

    mrae_loss, rmse_loss, psnr_loss = utils.validate(val_loader, model, mrae, rmse, psnr,
                                                     sample_folder=output_dir, is_save=opt.save)
    print('MRAE:{%.4f}, RMSE: {%.4f}, PNSR:{%.2f}' % (mrae_loss, rmse_loss, psnr_loss))
