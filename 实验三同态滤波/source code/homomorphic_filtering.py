import os
import cv2
import math
import numpy as np


def homomorphic_filtering(img, rl=0.5, rh=1.0, D0=20, c=1):
    # Convert the image to float32 for further processing
    image = np.float32(img)
    new_img = np.zeros_like(img, dtype="uint8")
    img_shape = image.shape
    mask = np.zeros((img_shape[0], img_shape[1]), dtype="float64")
    center_row, center_col = int(img_shape[0]/2), int(img_shape[1]/2)

    # Take the natural logarithm of the image and normalize
    log_image = np.log1p(image)
    log_image = log_image / np.log(256)
    
    # Create mask
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            dis = math.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            mask[i, j] =(1-np.exp(-(c ** 2)*(dis ** 2)/(2 * D0 ** 2)))*(rh-rl)+rl

    for m in range(img_shape[2]):
        # Perform the Fourier Transform
        dft = np.fft.fft2(log_image[:, :, m])

        # Shift the zero frequency components to the center
        dft_shift = np.fft.fftshift(dft)

        filtered_dft = dft_shift*mask
        # Shift the zero frequency components back to the corners
        filtered_dft_shift = np.fft.ifftshift(filtered_dft)

        # Perform the inverse Fourier Transform
        filtered_image = np.fft.ifft2(filtered_dft_shift)
        filtered_image = np.array(abs(filtered_image))
        filtered_image = np.exp(filtered_image) - 1

        # Normalize the filtered image to the range [0, 255]
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
        filtered_image = np.uint8(np.clip(filtered_image, 0, 255))
        new_img[:, :, m] = filtered_image
    
    return new_img

