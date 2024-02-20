import cv2
import numpy as np
import h5py


def get_crop_img_from_dataset(data_root, patch_size):
    bgrs = []
    bgr_data_path = data_root + '\\Valid_RGB\\'
    valid_list_path = data_root + '\\split_txt\\valid_list.txt'

    with open(valid_list_path, 'r') as fin:
        bgr_list = [line.replace('\n', '.jpg') for line in fin]
    bgr_list.sort()
    h, w = 482, 512
    patch_per_line = (w - patch_size) // 8 + 1
    patch_per_col = (h - patch_size) // 8 + 1
    patch_per_img = patch_per_line * patch_per_col

    for i in range(len(bgr_list)):
        bgr_path = bgr_data_path + bgr_list[i]
        bgr = cv2.imread(bgr_path)
        bgrs.append(bgr)

    for i in range(50):
        img_idx, patch_idx = i // patch_per_img, i % patch_per_img
        h_idx, w_idx = patch_idx // patch_per_line, patch_idx % patch_per_line
        crop_bgr = bgrs[img_idx][h_idx * 8:h_idx * 8 + patch_size, w_idx * 8:w_idx * 8 + patch_size, :]
        cv2.imwrite(f'D:\Datasets\\rgbs\crop_bgr_{i}.jpg', crop_bgr)


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
    _, h, w = hyper.shape
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
    # get_crop_img_from_dataset(r'D:\Datasets\data2022', 128)
    # get_crop_img_from_single_img(r'D:\Datasets\data2022\ARAD_1K_0914.jpg', 128)
    get_crop_mat_from_single_mat(r'D:\Datasets\data2022\ARAD_1K_0914.mat', 128)
