import os 
import cv2
import math
import numpy as np


class LowpassFilter():
    def __init__(self) -> None:
        pass

    def ideal_lowpass_filter(self, img, D0):
        rows, cols=img.shape
        center_row, center_col=int(rows/2), int(cols/2) 
        mask=np.zeros((rows,cols), dtype="uint8") 
        for i in range(rows):
            for j in range(cols):
                dis = math.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                if dis <= D0:
                    mask[i,j]=1
                else:
                    mask[i,j]=0
        return mask * img
    
    def gaussian_lowpass_filter(self, img, D0):
        rows, cols=img.shape
        center_row, center_col=int(rows/2),int(cols/2)
        mask=np.zeros((rows,cols), dtype="float64") 
        for i in range(rows):
            for j in range(cols):
                dis = math.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                mask[i, j] =np.exp(-(dis ** 2)/(2 * D0 ** 2))
        return mask * img
    



    

