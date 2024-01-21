import cv2
import numpy as np


def erodeOperation(img, iter=1):
    es = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_img = cv2.erode(img, es, iterations=iter)
    return new_img

def dilateOperation(img, iter=1):
    es = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_img = cv2.dilate(img, es, iterations=iter)
    return new_img

def openOperation(img):
    es = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, es)
    return new_image

def closeOperation(img):
    es = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, es)
    return new_image


##############Enhance the image in other way#######################
class breakpointEnhance:
    def __init__(self) -> None:
        pass

    def alphaTrimmedMeanFilter(self, image, d, m, n):
        imagePadded = np.pad(image, ((m//2, m//2), (n//2, n//2)), mode='edge')
        new_image = image.copy().astype("float64")
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                subimage = np.sort(imagePadded[r:r + m, c:c + n].flatten(), axis=0)
                new_image[r, c] = np.sum(subimage[d//2:m*n-d//2])/(subimage[d//2:m*n-d//2].shape[0])
        return np.uint8(cv2.normalize(new_image, None, 0, 255, cv2.NORM_MINMAX))
    
    def neighbours(self, x, y, image):
        img = image
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

    def sum_number(self, n):
        S = 0
        for i in range(len(n)):
            if (n[i] == 255):
                S += 1
        return S

    def transitions_num(self, neighbours):
        n = neighbours +neighbours[0:1]
        S = 0
        for i in range(len(n)-1):
            if ((n[i],n[i+1]) == (0,255)):
                S += 1
                # print(S)
        return S

    # Zhang-Suen 
    def zhangSuen(self, image):
        Image_Thinned = image.copy()  # Making copy to protect original image
        changing1 = changing2 = 1
        while changing1 or changing2:  # Iterates until no further changes occur in the image
            # Step 1
            changing1 = []
            rows, columns = Image_Thinned.shape
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x][y] == 255 and  # Condition 0: Point P1 in the object regions
                            2 <= self.sum_number(n)<= 6 and  # Condition 1: 2<= N(P1) <= 6
                            self.transitions_num(n) == 1 and  # Condition 2: S(P1)=1
                            P2 * P4 * P6 == 0 and  # Condition 3
                            P4 * P6 * P8 == 0):  # Condition 4
                        changing1.append((x, y))
            for x, y in changing1:
                Image_Thinned[x][y] = 0
            # Step 2
            changing2 = []
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x][y] == 255 and  # Condition 0
                            2 <= self.sum_number(n) <= 6 and  # Condition 1
                            self.transitions_num(n) == 1 and  # Condition 2
                            P2 * P4 * P8 == 0 and  # Condition 3
                            P2 * P6 * P8 == 0):  # Condition 4
                        changing2.append((x, y))
            for x, y in changing2:
                Image_Thinned[x][y] 
        return Image_Thinned
    
    def fingerEnhance(self, img):
        new_img = img.copy()
        new_img = self.alphaTrimmedMeanFilter(new_img, 20, 5, 5)
        contours, _ = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_img = cv2.drawContours(new_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        ske_img = self.zhangSuen(new_img)
        return new_img, ske_img
    
