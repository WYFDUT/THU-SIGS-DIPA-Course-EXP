import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import calGlcm as cal
import hu


def display_co_occurrence_matrix(matrix):
    plt.imshow(matrix, cmap='gray')
    plt.title('Co-occurrence Matrix')
    plt.colorbar()
    plt.show()

def img_show(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    imgpath = "C:\\Users\\WYF\\Desktop\\G1.bmp"
    img = cv2.imread(imgpath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_show(img_gray)

    res = cal.calculate_cooccurrence_matrix(img_gray, distance=1, angles=[0])
    #from skimage.feature import graycomatrix
    #co_matrix = graycomatrix(img_gray, [1], [3*np.pi/4], levels=img_gray.max()+1, symmetric=True, normed=True)
    #display_co_occurrence_matrix(co_matrix[:, :, 0, 0])
    display_co_occurrence_matrix(res)
    #print(cal.calculate_texture_features(co_matrix[:, :, 0, 0]))
    print(cal.calculate_texture_features(res))
    
    
    imgpath2 = "C:\\Users\\WYF\\Desktop\\M1.bmp"
    img2 = cv2.imread(imgpath2)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_show(img_gray2)

    ans = hu.calculate_hu_moments(img_gray2)
    print(ans)
