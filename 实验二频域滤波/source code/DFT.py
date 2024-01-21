import os 
import cv2
import math
import numpy as np


class DiscreteFourierTrans():
    def __init__(self) -> None:
        pass
    
    def fourier_transform_2d(self, image, shift=True):
        '''
        Discrete space fourier transform
        x: Input matrix
        '''
        N1, N2 = image.shape
        X = np.zeros((N1, N2), dtype=np.complex64)
        n1, n2 = np.mgrid[0:N1, 0:N2]

        for w1 in range(N1):
            for w2 in range(N2):
                j2pi = np.zeros((N1, N2), dtype=np.complex64)
                j2pi.imag = 2*np.pi*(w1*n1/N1 + w2*n2/N2)
                X[w1, w2] = np.sum(image*np.exp(-j2pi))
        if shift:
            X = np.roll(X, N1//2, axis=0)
            X = np.roll(X, N2//2, axis=1)
        return X

    def inverse_fourier_transform_2d(self, fourier_image, shift=True):
        '''
        Inverse discrete space fourier transform
        X: Complex matrix
        '''
        N1, N2 = fourier_image.shape
        x = np.zeros((N1, N2))
        k1, k2 = np.mgrid[0:N1, 0:N2]
        if shift:
            fourier_image = np.roll(fourier_image, -N1//2, axis=0)
            fourier_image = np.roll(fourier_image, -N2//2, axis=1)
        for n1 in range(N1):
            for n2 in range(N2):
                j2pi = np.zeros((N1, N2), dtype=np.complex64)
                j2pi.imag = 2*np.pi*(n1*k1/N1 + n2*k2/N2)
                x[n1, n2] = abs(np.sum(fourier_image*np.exp(j2pi)))
        return 1/(N1*N2)*x

def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

