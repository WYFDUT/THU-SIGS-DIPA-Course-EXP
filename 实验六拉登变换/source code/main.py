import cv2
import numpy as np
import phantominator
import matplotlib.pyplot as plt
import reconstruction_from_projections as RFP

from phantominator import shepp_logan


if __name__ == "__main__":
    """
    ph_img = np.uint8(shepp_logan(512)*255)
    print(ph_img.shape)
    cv2.imshow("test", ph_img)
    #cv2.imshow("test", np.uint8(cv2.normalize(ph_img, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("shepp_logan.jpg", np.uint8(ph_img))

    dst = RFP.radon_transform(ph_img, ph_img.shape[1])
    cv2.imshow("test", np.uint8(cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("radon.jpg", np.uint8(cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)))
    """
    ph_img = cv2.cvtColor(cv2.imread("shepp_logan.jpg"), cv2.COLOR_BGR2GRAY)
    #dst = cv2.cvtColor(cv2.imread("radon.jpg"), cv2.COLOR_BGR2GRAY)
    """
    iradon_img = RFP.inverse_radon_transform(dst, ph_img.shape[1])
    cv2.imshow("test", np.uint8(cv2.normalize(iradon_img, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    dst = RFP.radon_transform(ph_img, ph_img.shape[1]//4)
    iradon_img_win = RFP.inverse_filter_radon_transform(dst, ph_img.shape[1]//4, window='hanning')
    cv2.imshow("test", np.uint8(cv2.normalize(iradon_img_win, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


