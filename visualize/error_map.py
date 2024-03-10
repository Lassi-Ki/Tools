import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


sorted_img_list = [
    r"D:\Codes\Tools\output\out_splits\hscnn_plus_split",
    r"D:\Codes\Tools\output\out_splits\awan_split",
    r"D:\Codes\Tools\output\out_splits\hrnet_split",
    r"D:\Codes\Tools\output\out_splits\mst_split",
    r"D:\codes\Tools\output\out_splits\ms3t_split",
    r"D:\codes\Tools\output\out_splits\mst_plus_plus_split",
    r"D:\codes\Tools\output\out_splits\gdlnet_split",
]


def make_error_map(gen_img_path, real_img_path):
    gen_img = cv2.imread(gen_img_path, cv2.IMREAD_GRAYSCALE)
    real_img = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)

    diff_image = np.abs(gen_img.astype(int) - real_img.astype(int)).astype(np.uint8)

    plt.imshow(diff_image, cmap='jet')
    plt.axis('off')
    plt.title('Error Map')
    plt.colorbar()
    plt.show()


def make_group_error_map(gen_folder, real_img_path, patch, channel, save_path=None):
    # gen_folder_list = os.listdir(gen_folder)
    gen_folder_list = sorted_img_list
    _gen_folder_list = []

    for path in gen_folder_list:
        # gen_img_path = os.path.join(gen_folder, path)
        gen_img_path = path
        # 增加具体哪个 patch 哪个 channel 的后缀
        gen_img_path = os.path.join(gen_img_path,
                                    f'patch_crop_bgr_{patch}',
                                    f'crop_bgr_{patch}_channel_{channel}.jpg')
        _gen_folder_list.append(gen_img_path)

    gen_folder_list = _gen_folder_list
    # print(gen_folder_list)
    real_img_path = os.path.join(real_img_path,
                                 f'patch_crop_hyper_{patch}',
                                 f'crop_hyper_{patch}_channel_{channel}.jpg')
    # 读取基准图片
    base_image = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)

    # 创建 Matplotlib 画框
    plt.figure(figsize=(10, 8))

    for idx, current_img_path in enumerate(gen_folder_list):
        current_model = current_img_path.split('\\')[-3].rsplit('_split')[0]
        current_img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
        diff_image = np.abs(current_img.astype(int) - base_image.astype(int)).astype(np.uint8)
        # 将误差图添加到 subplot 中
        plt.subplot(2, 4, idx+1)
        plt.imshow(diff_image, cmap='jet')
        plt.axis('off')
        plt.title(f'{current_model}')

    # 增加原图
    plt.subplot(2, 4, 8)
    plt.imshow(base_image, cmap='gray')
    plt.axis('off')
    plt.title('Real')

    # 调整 subplot 之间的间距
    plt.tight_layout()
    # 保存画框
    save_path = os.path.join(save_path, f'patch_{patch}_channel_{channel}.png')
    plt.savefig(save_path)
    plt.close()


def traverse():
    for patch_th in range(96, 120):
        for channel_th in range(0, 31):
            make_group_error_map("",
                                 r"D:\codes\Tools\output\HSIs_split",
                                 patch_th, channel_th,
                                 r"D:\Codes\Tools\output\error_map")
            # print(f'patch: {patch_th}, channel: {channel_th} sueecssfully!')
        print(f'patch: {patch_th} sueecssfully!')
        if patch_th == 107:
            break


def make_single_error_map(real_img_path, patch, channel, save_path=None):
    save_path = os.path.join(save_path, f'patch_{patch}_channel_{channel}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gen_folder_list = sorted_img_list
    _gen_folder_list = []

    for path in gen_folder_list:
        gen_img_path = path
        gen_img_path = os.path.join(gen_img_path,
                                    f'patch_crop_bgr_{patch}',
                                    f'crop_bgr_{patch}_channel_{channel}.jpg')
        _gen_folder_list.append(gen_img_path)
    gen_folder_list = _gen_folder_list

    real_img_path = os.path.join(real_img_path,
                                 f'patch_crop_hyper_{patch}',
                                 f'crop_hyper_{patch}_channel_{channel}.jpg')
    base_image = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(save_path, 'real.jpg'), base_image)

    for idx, current_img_path in enumerate(gen_folder_list):
        current_model = current_img_path.split('\\')[-3].rsplit('_split')[0]
        current_img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
        diff_image = np.abs(current_img.astype(int) - base_image.astype(int)).astype(np.uint8)
        normalized_gray_image = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)
        color_img = cv2.applyColorMap(normalized_gray_image, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, f'{current_model}.jpg'), color_img)


if __name__ == "__main__":
    # traverse()
    make_single_error_map(r"D:\codes\Tools\output\HSIs_split", 104, 0, r"D:\Codes\Tools\output\error_map")
