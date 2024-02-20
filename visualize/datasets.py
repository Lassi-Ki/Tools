from torch.utils.data import Dataset
import numpy as np
import cv2
import h5py


class NTIRE2022Dataset(Dataset):
    def __init__(self, data_root, patch_size, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        self.patch_size = patch_size
        hyper_data_path = data_root + 'Valid_spectral'
        bgr_data_path = data_root + 'Valid_RGB'
        valid_list_path = data_root + 'split_txt/valid_list.txt'

        with open(valid_list_path, 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()

        h, w = 482, 512
        self.patch_per_line = (w - patch_size) // 8 + 1
        self.patch_per_col = (h - patch_size) // 8 + 1
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

    def __getitem__(self, idx):
        # TODO: stride = 128
        stride = 8
        crop_size = self.patch_size
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
        bgr = self.bgrs[img_idx][:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        hyper = self.hypers[img_idx][:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]

        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers * self.patch_per_img)
