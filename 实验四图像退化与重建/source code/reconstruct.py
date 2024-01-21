import cv2
import numpy as np
import degraded
import matplotlib.pyplot as plt


def wienerFilter(input, H_func, K=0.01):  
    fftImg = np.fft.fft2(input)
    
    fftWiener = np.conj(H_func) / (np.abs(H_func)**2 + K)
    imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
    imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
    return imgWienerFilter

def wienerFilter2(input, H_func, K=0.01):  
    fftImg = np.fft.fftshift(np.fft.fft2(input))
    
    fftWiener = np.conj(H_func) / (np.abs(H_func)**2 + K)
    imgWienerFilter = np.fft.ifft2(np.fft.ifftshift(fftImg * fftWiener))
    imgWienerFilter = np.abs(imgWienerFilter)
    return imgWienerFilter

def optimumNotchFilter(input, noisy):
    a = 5  # Neighborhood size, horizontal
    b = 5  # Neighborhood size, vertical
    gxy = input  # Original image
    fxyHat = np.zeros(gxy.shape)  # Optimum Notch Filtered result image
    wxy = np.zeros(gxy.shape)  # Store weights for each pixel coordinate

    # Padding the original image and noise image with zeros
    gxyPadded = np.pad(gxy, ((b, b), (a, a)), mode='constant')
    ETAxyPadded = np.pad(noisy, ((b, b), (a, a)), mode='constant')

    for r in range(fxyHat.shape[0]):
        for c in range(fxyHat.shape[1]):
            subimage_gxy = gxyPadded[r:r + 2 * b + 1, c:c + 2 * a + 1]
            subimage_ETAxy = ETAxyPadded[r:r + 2 * b + 1, c:c + 2 * a + 1]

            wxy[r, c] = (np.mean(subimage_gxy * subimage_ETAxy) -
                np.mean(subimage_gxy) * np.mean(subimage_ETAxy)) / (
                np.mean(subimage_ETAxy ** 2) - np.mean(subimage_ETAxy) ** 2)
            
            fxyHat[r, c] = gxy[r, c] - wxy[r, c] * noisy[r, c]
    return fxyHat
