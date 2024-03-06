import cv2
import numpy as np
import h5py


def get_crop_img_from_dataset(data_root, patch_size):
    bgrs = []
    hypers = []
    bgr_data_path = data_root + '\\Valid_RGB\\'
    hyper_data_path = data_root + '\\Valid_spectral\\'
    valid_list_path = data_root + '\\split_txt\\valid_list.txt'

    with open(valid_list_path, 'r') as fin:
        bgr_list = [line.replace('\n', '.jpg') for line in fin]
        hyper_list = [line.replace('jpg', 'mat') for line in bgr_list]
    bgr_list.sort()
    hyper_list.sort()

    patch_per_line = 4
    patch_per_col = 3
    patch_per_img = patch_per_line * patch_per_col

    for i in range(len(bgr_list)):
        bgr_path = bgr_data_path + bgr_list[i]
        bgr = cv2.imread(bgr_path)
        # bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # bgr = np.transpose(bgr, [2, 0, 1])
        bgrs.append(bgr)    # 482, 512, 3

        hyper_path = hyper_data_path + hyper_list[i]
        with h5py.File(hyper_path, 'r') as mat:
            hyper = np.float32(np.array(mat['cube']))   # 31, 512, 482
        # hyper = np.transpose(hyper, [0, 2, 1])
        hypers.append(hyper)

    for idx in range(120):
        img_idx = idx // patch_per_img
        patch_idx = idx % patch_per_img

        bgr = bgrs[img_idx]
        hyper = hypers[img_idx]

        h = patch_idx // patch_per_line
        w = patch_idx % patch_per_line

        # bgr = bgr[:, h * 128:h * 128 + patch_size, w * 128:w * 128 + patch_size]
        # hyper = hyper[:, h * 128:h * 128 + patch_size, w * 128:w * 128 + patch_size]
        bgr = bgr[h * 128:h * 128 + patch_size, w * 128:w * 128 + patch_size, :]
        hyper = hyper[:, w * 128:w * 128 + patch_size, h * 128:h * 128 + patch_size]

        cv2.imwrite(f'D:\Codes\Tools\output\RGBs\crop_bgr_{idx}.jpg', bgr)
        with h5py.File(f'D:\Codes\Tools\output\HSIs\crop_hyper_{idx}.mat', 'w') as f:
            f.create_dataset('cube', data=hyper)
        print(f'crop_{idx} successfully!')


def get_crop_img_from_single_img(data_root, patch_size):
    bgr = cv2.imread(data_root)
    h, w, _ = bgr.shape
    patch_per_line = (w - patch_size) // 128 + 1
    patch_per_col = (h - patch_size) // 128 + 1
    patch_per_img = patch_per_line * patch_per_col

    for i in range(patch_per_img):
        h_idx, w_idx = i // patch_per_line, i % patch_per_line
        if h_idx * 128 + patch_size > h or w_idx * 128 + patch_size > w:
            continue
        crop_bgr = bgr[h_idx * 128:h_idx * 128 + patch_size, w_idx * 128:w_idx * 128 + patch_size, :]
        cv2.imwrite(f'D:\Datasets\\rgbs_0914\crop_bgr_{i}.jpg', crop_bgr)


def get_crop_mat_from_single_mat(data_root, patch_size):
    # (31, 512, 482)
    with h5py.File(data_root, 'r') as f:
        hyper = np.float32(np.array(f['cube']))
    _, w, h = hyper.shape
    hyper = np.transpose(hyper, [2, 1, 0])
    patch_per_line = (w - patch_size) // 128 + 1
    patch_per_col = (h - patch_size) // 128 + 1
    patch_per_img = patch_per_line * patch_per_col

    for i in range(patch_per_img):
        h_idx, w_idx = i // patch_per_line, i % patch_per_line
        if h_idx * 128 + patch_size > h or w_idx * 128 + patch_size > w:
            continue
        crop_hyper = hyper[h_idx * 128:h_idx * 128 + patch_size, w_idx * 128:w_idx * 128 + patch_size, :]
        crop_hyper = np.transpose(crop_hyper, [2, 0, 1])
        with h5py.File(f'D:\Datasets\\hypers_0914\crop_hyper_{i}.mat', 'w') as f:
            f.create_dataset('cube', data=crop_hyper)


if __name__ == '__main__':
    get_crop_img_from_dataset(r'D:\Datasets\data2022test\\', 128)
    # get_crop_img_from_single_img(r'D:\Datasets\data2022\ARAD_1K_0914.jpg', 128)
    # get_crop_mat_from_single_mat(r'D:\Datasets\data2022\ARAD_1K_0914.mat', 128)
