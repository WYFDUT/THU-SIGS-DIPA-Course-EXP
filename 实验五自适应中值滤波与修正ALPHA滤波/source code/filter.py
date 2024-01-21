import cv2
import numpy as np
import matplotlib.pyplot as plt


def alphaTrimmedMeanFilter(image, d, m, n):

    imagePadded = np.pad(image, ((m//2, m//2), (n//2, n//2)), mode='edge')
    new_image = image.copy().astype("float64")

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            subimage = np.sort(imagePadded[r:r + m, c:c + n].flatten(), axis=0)
            new_image[r, c] = np.sum(subimage[d//2:m*n-d//2])/(subimage[d//2:m*n-d//2].shape[0])

    cv2.imshow("Turbulence Blur", np.uint8(cv2.normalize(new_image, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.uint8(cv2.normalize(new_image, None, 0, 255, cv2.NORM_MINMAX))


def adaptiveMedianFilter(image, d0, dmax):
    hPad, wPad = dmax//2, dmax//2
    imagePadded = np.pad(image.copy(), ((dmax//2, dmax//2), (dmax//2, dmax//2)), mode='edge')
    imgAdaMedFilter = np.zeros(image.shape)  
    for i in range(hPad, hPad+image.shape[0]):
        for j in range(wPad, wPad+image.shape[1]):
            ksize = d0
            k = int(ksize/2)
            pad = imagePadded[i-k:i+k+1, j-k:j+k+1]
            zxy = image[i-hPad][j-wPad]
            zmin = np.min(pad)
            zmed = np.sort(pad.flatten(), axis=0)[pad.flatten().shape[0]//2]
            zmax = np.max(pad)

            if zmin < zmed < zmax:
                if zmin < zxy < zmax:
                    imgAdaMedFilter[i-hPad, j-wPad] = zxy
                else:
                    imgAdaMedFilter[i-hPad, j-wPad] = zmed
            else:
                while True:
                    ksize = ksize + 2
                    if zmin < zmed < zmax or ksize > dmax:
                        break
                    k = int(ksize / 2)
                    pad = imagePadded[i-k:i+k+1, j-k:j+k+1]
                    zmed = np.sort(pad.flatten(), axis=0)[pad.flatten().shape[0]//2]
                    zmin = np.min(pad)
                    zmax = np.max(pad)
                if zmin < zmed < zmax or ksize > dmax:
                    if zmin < zxy < zmax:
                        imgAdaMedFilter[i-hPad, j-wPad] = zxy
                    else:
                        imgAdaMedFilter[i-hPad, j-wPad] = zmed
    cv2.imshow("Turbulence Blur", np.uint8(cv2.normalize(imgAdaMedFilter, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.uint8(cv2.normalize(imgAdaMedFilter, None, 0, 255, cv2.NORM_MINMAX))


def saltpepper_noise(image, proportion):
    image_copy = image.copy()
    img_Y, img_X = image.shape
    X = np.random.randint(img_X,size=(int(proportion*img_X*img_Y),))
    Y = np.random.randint(img_Y,size=(int(proportion*img_X*img_Y),))
    image_copy[Y, X] = np.random.choice([0, 255], size=(int(proportion*img_X*img_Y),))
    return image_copy


def gaussian_noise(img, mean, sigma):
    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    return gaussian_out

            

