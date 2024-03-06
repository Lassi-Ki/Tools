from torch.utils.data import Dataset
import numpy as np
import cv2
import h5py


class NTIRE2022Dataset(Dataset):
    def __init__(self, opt, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        self.crop_size = opt.patch_size
        self.stride = opt.stride

        hyper_data_path = opt.dataset_path + 'Valid_spectral'
        bgr_data_path = opt.dataset_path + 'Valid_RGB'
        valid_list_path = opt.dataset_path + 'split_txt/valid_list.txt'

        with open(valid_list_path, 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()

        self.patch_per_line = 4
        self.patch_per_col = 3
        self.patch_per_img = self.patch_per_line * self.patch_per_col

        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])

            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()

        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size

        img_idx = idx // self.patch_per_img
        patch_idx = idx % self.patch_per_img

        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]

        h = patch_idx // self.patch_per_line
        w = patch_idx % self.patch_per_line

        bgr = bgr[:, h * stride: h * stride + crop_size, w * stride: w * stride + crop_size]
        hyper = hyper[:, h * stride: h * stride + crop_size, w * stride: w * stride + crop_size]

        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.length
