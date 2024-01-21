import numpy as np


def calculate_moments(image):
    image = image.copy().astype("float64")
    x, y = np.meshgrid(np.arange(image.shape[1]).astype("float64"), np.arange(image.shape[0]).astype("float64"), indexing='xy')

    # 计算零阶矩
    m00 = np.sum(image)

    # 计算一阶矩
    m10 = np.sum(x*image)
    m01 = np.sum(y*image)

    # 计算中心矩
    x_bar = m10 / m00
    y_bar = m01 / m00

    # 计算二阶矩
    m20 = np.sum((x - x_bar) ** 2 * image)
    m02 = np.sum((y - y_bar) ** 2 * image)
    m11 = np.sum((x - x_bar) * (y - y_bar) * image)

    # 计算三阶矩
    m30 = np.sum((x - x_bar) ** 3 * image)
    m12 = np.sum((x - x_bar) * (y - y_bar) ** 2 * image)
    m21 = np.sum((x - x_bar) ** 2 * (y - y_bar) * image)
    m03 = np.sum((y - y_bar) ** 3 * image)

    return (m00, m10, m01, m20, m02, m11, m30, m12, m21, m03)

def calculate_hu_moments(image):
    moments = calculate_moments(image)
    #print(moments)
    mu20, mu02, mu11, mu30, mu12, mu21, mu03 = moments[3:10]

    nu20 = mu20 / moments[0] ** 2
    nu02 = mu02 / moments[0] ** 2
    nu11 = mu11 / moments[0] ** 2
    nu30 = mu30 / moments[0] ** 2.5
    nu12 = mu12 / moments[0] ** 2.5
    nu21 = mu21 / moments[0] ** 2.5
    nu03 = mu03 / moments[0] ** 2.5

    # 计算7阶Hu不变矩
    hu1 = func(nu20 + nu02)
    hu2 = func((nu20 - nu02) ** 2 + 4 * nu11 ** 2)
    hu3 = func((nu30 - 3 * nu12) ** 2 + (3 * nu21 - nu03) ** 2)
    hu4 = func((nu30 + nu12) ** 2 + (nu21 + nu03) ** 2)
    hu5 = func((nu30 - 3 * nu12) * (nu30 + nu12) * ((nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2) + (3 * nu21 - nu03) * (nu21 + nu03) * (3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2))
    hu6 = func((nu20 - nu02) * ((nu30 + nu12) ** 2 - (nu21 + nu03) ** 2) + 4 * nu11 * (nu30 + nu12) * (nu21 + nu03))
    hu7 = func((3 * nu21 - nu03) * (nu30 + nu12) * ((nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2) - (nu30 - 3 * nu12) * (nu21 + nu03) * (3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2))

    return hu1, hu2, hu3, hu4, hu5, hu6, hu7

def func(x):
    return -np.sign(x)*np.log10(np.abs(x))

