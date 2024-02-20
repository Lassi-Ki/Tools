import numpy as np
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt


def view_mat_info_by_scipy(mat_file):
    mat = sio.loadmat(mat_file)
    print(mat.keys())
    print(mat['HSI'].shape)
    print("------------------------------")
    hyper = np.float32(np.array(mat['HSI'])).flatten()
    print("数组的最小值:", np.min(hyper))
    print("数组的最大值:", np.max(hyper))
    plt.hist(hyper, bins=30, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.show()


def view_mat_info_by_h5py(mat_file):
    with h5py.File(mat_file, 'r') as f:
        hyper = np.float32(np.array(f['cube']))
        print(hyper.shape)
        hyper = hyper.flatten()
        # hyper = np.transpose(hyper, [0, 2, 1])
        print(f.keys())
        print(f'Max: {hyper.max()}, Min: {hyper.min()}')
        print(f'Shape: {hyper.shape}')
        plt.hist(hyper, bins=30, edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Data')
        plt.show()


if __name__ == '__main__':
    # mat_file = r'D:\Datasets\CAVE\Train_spectral\balloons_ms.mat'
    mat_file = r'D:\Datasets\hypers_0914\crop_hyper_0.mat'
    # mat_file = './datas/fake_and_real_peppers_ms.mat'
    # view_mat_info_by_scipy(mat_file)
    view_mat_info_by_h5py(mat_file)
