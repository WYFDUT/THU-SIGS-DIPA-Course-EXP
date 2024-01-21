import cv2
import numpy as np
import opening_and_closing as OAC
import fingerprint_enhancer


def imgShow(img, name='test'):
    cv2.imshow(name, np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test1 = OAC.breakpointEnhance()
    img = cv2.cvtColor(cv2.imread("fingerprint.tif"), cv2.COLOR_BGR2GRAY)
    imgShow(img)

    # Open & Dilate
    #new_img = OAC.openOperation(img)
    #new_img = OAC.dilateOperation(new_img)
    #imgShow(new_img)

    # Opne & Close
    #new_img = OAC.openOperation(img)
    #new_img = OAC.closeOperation(new_img)
    #imgShow(new_img)

    # Own method
    #_, new_img = test1.fingerEnhance(img)
    #imgShow(new_img)

    # Gabor
    #new_img = test1.alphaTrimmedMeanFilter(img, 16, 5, 5)
    new_img = img
    new_img = cv2.copyMakeBorder(new_img,20,20,20,20,borderType=cv2.BORDER_CONSTANT,value=0)
    out = fingerprint_enhancer.enhance_Fingerprint(new_img)
    print(new_img.shape, out.shape)
    imgShow(out)


