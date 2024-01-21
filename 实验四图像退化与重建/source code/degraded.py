import cv2
import numpy as np
import matplotlib.pyplot as plt


def turbulenceBlur(img, k=0.001):
    # Atmospheric turbulence model
    M, N = img.shape[:2]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    radius = np.sqrt((u-M//2)**2 + (v-N//2)**2)
    kernel = np.exp(-k*np.power((radius**2), 5/6))
    return kernel

def getMotionDsf(shape, dist, eps, a=0.1, b=0.1):
    # Motion Blurred model
    xCenter = (shape[0] - 1) / 2
    yCenter = (shape[1] - 1) / 2
    sinVal = np.sin(a * np.pi)
    cosVal = np.cos(b * np.pi)
    PSF = np.zeros(shape)  # Point diffusion func
    for i in range(dist): 
        xOffset = round(sinVal * i)
        yOffset = round(cosVal * i)
        PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
    return np.fft.fft2(PSF / PSF.sum()) + eps





