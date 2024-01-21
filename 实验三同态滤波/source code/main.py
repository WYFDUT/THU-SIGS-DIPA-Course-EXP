import os
import cv2
import fnmatch
import homomorphic_filtering
import matplotlib.pyplot as plt


if __name__ == "__main__":

    img_path = "C:\\Users\\WYF\\Desktop\\DIP\\Homomorphic_Filtering"

    imgs = []

    imgfile_rootpath = [name for name in os.listdir(img_path)]
    for index, img_name in enumerate(imgfile_rootpath):
        if fnmatch.fnmatch(img_name, '*.jpg') or fnmatch.fnmatch(img_name, '*.png') or fnmatch.fnmatch(img_name, '*.bmp'):
            #imgs_name.append(img_name)
            img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_COLOR)
            new_img = homomorphic_filtering.homomorphic_filtering(img)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            plt.figure(1)
            plt.imshow(new_img)
            plt.show()
            imgs.append(img)
        else:
            break


