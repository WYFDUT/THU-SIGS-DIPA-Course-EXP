import cv2
import filter as F
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    img_path = "C:\\Users\\WYF\\Desktop\\DIP\\Experiments2_Filtering\\A_SP_1.bmp"

    image = cv2.imread(img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = F.gaussian_noise(image, 0, 0.05)
    image = F.saltpepper_noise(image, 0.5)

    
    image = F.adaptiveMedianFilter(image, 3, 9)
    image = F.alphaTrimmedMeanFilter(image, 16, 5, 5)
    
    
