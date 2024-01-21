import numpy as np
import cv2
import math


"""
def cal_Glcm(img, d_x=1, d_y=0, normed=True):
    tmp = img.copy()
    res = np.zeros((np.max(img)+1, np.max(img)+1)).astype("float32")
    print(img.shape)
    h, w = img.shape
    for j in range(h - d_y):
        for i in range(0, w - d_x, 1):
            res[tmp[j][i], tmp[j+d_y][i+d_x]] += 1

    if normed:
        res = res / np.sum(res)

    #return np.uint8(cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX))
    return res
"""


def calculate_cooccurrence_matrix(image, distance=1, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], normed=True):
    # Convert the image to grayscale if it is a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the co-occurrence matrix
    max_pixel_value = image.max()
    cooccurrence_matrix = np.zeros((max_pixel_value+1, max_pixel_value+1)).astype("float32")
    

    # Iterate through each pixel in the image
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            # Iterate through each angle
            for angle in angles:
                # Calculate the neighbor pixel coordinates
                dx, dy = distance * np.cos(angle), distance * np.sin(angle)
                x_offset = int(i + dx)
                y_offset = int(j + dy)

                # Check if the neighbor coordinates are within the image boundaries
                if 0 <= x_offset < image.shape[1] and 0 <= y_offset < image.shape[0]:
                    # Increment the co-occurrence matrix entry for the pixel pair
                    cooccurrence_matrix[image[j, i], image[y_offset, x_offset]] += 1

    #return np.uint8(cv2.normalize(cooccurrence_matrix, None, 0, 255, cv2.NORM_MINMAX))
    if normed:
        cooccurrence_matrix /= np.sum(cooccurrence_matrix)
    return cooccurrence_matrix


def calculate_texture_features(co_matrix):
    # 最大概率
    max_probability = np.max(co_matrix)

    # 创建网格坐标
    i, j = np.meshgrid(np.arange(co_matrix.shape[0]), np.arange(co_matrix.shape[1]), indexing='ij')

    # 相关性、对比度、一致性、同质性、同质性能量、熵
    contrast = np.sum((i - j) ** 2 * co_matrix)
    
    correlation = cal_correlation(co_matrix)

    homogeneity = np.sum(co_matrix / (1 + np.abs(i - j)))
    homogenous_energy = np.sum(co_matrix ** 2)
    entropy = -np.sum(co_matrix * np.log2(co_matrix + 1e-10))

    return max_probability, contrast, correlation, homogeneity, homogenous_energy, entropy


def cal_correlation(co_matrix):
    # Create meshgrid
    i, j = np.meshgrid(np.arange(co_matrix.shape[0]), np.arange(co_matrix.shape[1]), indexing='ij')

    # Calculate mean and standard deviation along rows and columns
    mean_i, mean_j = np.sum(i * co_matrix), np.sum(j * co_matrix)
    std_i, std_j = np.sqrt(np.sum((i - mean_i)**2 * co_matrix)), np.sqrt(np.sum((j - mean_j)**2 * co_matrix))

    # Calculate correlation
    correlation = np.sum(((i - mean_i) * (j - mean_j) * co_matrix)) / (std_i * std_j)
    return correlation