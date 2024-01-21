import cv2
import numpy as np
from scipy import ndimage

def radon_transform(img, steps):
    print(steps)
    dst = np.zeros((img.shape), dtype=np.float32)
    for i in range(steps):
        res = ndimage.rotate(img, -i * 180 / steps, reshape=False).astype(np.float32)
        dst[:, i] = sum(res)
    return dst


def inverse_radon_transform(image, steps):
    print(steps)
    channels = image.shape[0]
    dst = np.zeros((steps, channels, channels))
    for i in range(steps):
        temp = image[:, i]
        temp_expand_dim = np.expand_dims(temp, axis=0)
        temp_repeat = temp_expand_dim.repeat(channels, axis=0)
        dst[i] = ndimage.rotate(temp_repeat, i*180 / steps, reshape=False).astype(np.float64) 
        if i == 0 or (i*180 / steps) == 90 or (i*180 / steps)==135 or (i*180 / steps)==45:
            cv2.imshow("test", np.uint8(cv2.normalize(dst[i], None, 0, 255, cv2.NORM_MINMAX)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    iradon = np.sum(dst, axis=0)
    
    return iradon


def rl_filter(N, d):
    filterRL = np.zeros((N,))
    for i in range(N):
        filterRL[i] = - 1.0 / (np.power((i - N / 2) * np.pi * d, 2.0) + 1e-5) # 1e-5 加上一个不为零的小数，防止出现除0的问题
        if np.mod(i - N / 2, 2) == 0:
            filterRL[i] = 0
    filterRL[int(N/2)] = 1 / (4 * np.power(d, 2.0))
    return filterRL


def inverse_filter_radon_transform(image, steps, window="hamming"):
    channels = image.shape[0]
    origin = np.zeros((steps, channels, channels))
    filter = rl_filter(channels, 1)
    for i in range(steps):
        projectionValue = image[:, i]
        projectionValueFiltered = np.convolve(filter, projectionValue, "same")

        #u = np.hamming(projectionValueFiltered.shape[0])
        #projectionValueFiltered = projectionValueFiltered*u

        projectionValueExpandDim = np.expand_dims(projectionValueFiltered, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i*180/steps, reshape=False).astype(np.float64)

    if window == None:
        iradon = np.sum(origin, axis=0)
    else:
        window = window_func(image, win_type=window)
        iradon = np.sum(origin, axis=0)
        iradon = iradon * window
    return iradon 


def window_func(img, win_type='hamming'):
    M, N = img.shape[1], img.shape[0]
    if win_type == 'hamming':
        u = np.hamming(M)
        v = np.hamming(N)
    else:
        u = np.hanning(M)
        v = np.hanning(N)
    u, v = np.meshgrid(u, v)
    high_pass = np.sqrt(u**2 + v**2)
    kernel = high_pass
    return kernel


