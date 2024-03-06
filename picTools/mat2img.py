import hdf5storage as hdf5
import numpy as np
import os
import cv2


def split_images(mat_path, out_img_path):
    output_name = mat_path.split('\\')[-1] + '_split'
    split_path = os.path.join(out_img_path, output_name)
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    for filename in os.listdir(mat_path):
        spectral = hdf5.loadmat(os.path.join(mat_path, filename))   # 31 channels (31, 128, 128)
        spectral = (spectral['cube'] * 255).astype(np.uint8).transpose(2, 0, 1)
        subfile_path = os.path.join(split_path, f"patch_{filename.replace('.mat', '')}")
        if not os.path.exists(subfile_path):
            os.makedirs(subfile_path)
        for i in range(31):
            img = spectral[i, :, :]
            img_path = os.path.join(subfile_path, filename.replace('.mat', f"_channel_{i}.jpg"))
            if cv2.imwrite(img_path, img) is False:
                print(f"Failed to save {img_path}")
                return


if __name__ == '__main__':
    mat_path = r'D:\Codes\Tools\output\mst_plus_plus'
    out_img_path = r'D:\Codes\Tools\output'
    split_images(mat_path, out_img_path)
