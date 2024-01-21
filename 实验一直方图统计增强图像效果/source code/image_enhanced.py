import os 
import cv2
import numpy as np
import matplotlib.pylab as plt
from collections import defaultdict


class HistogramStatisticsEnhance:
    def __init__(self) -> None:
        pass

    def histogram_equalization(self, img):
        """
        @params img : input image
        @return new_img : output image after histogram equalization
        """
        h, w, c = img.shape
        channel_list = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
        new_img = np.zeros_like(img, dtype = np.uint8)

        # Each channel should do histogram equalization
        for channel_index in range(len(channel_list)):
            # build histogram dict 
            histogram = defaultdict(int)
            for i in range(256):
                # Each term of the histogram is normalized to the probability pr
                histogram[i] = np.sum(channel_list[channel_index]==i)/(h * w)

            # build mapping table for probability distributions
            prob = defaultdict(int)
            tmp, map_len = 0, len(list(histogram.keys()))
            for m in histogram.keys():
                tmp += histogram[m]
                prob[m] = int(map_len * tmp)

            for k in range(w):
                for l in range(h):
                    new_img[l, k, channel_index] = prob[channel_list[channel_index][l, k]]

        return new_img
    
    def adaptive_histogram_equalization(self, img, win_size):
        """
        @params img : input image
        @return new_img : output image after histogram equalization
        """
        h, w, c = img.shape
        new_img = np.zeros_like(img, dtype = np.uint8)
        # Padding operation
        img_resize = cv2.copyMakeBorder(img, win_size//2, win_size//2, win_size//2, win_size//2, borderType=cv2.BORDER_REFLECT)

        for i in range(h):
            for j in range(w):
                # Get window center point
                center_x, center_y = i + win_size // 2, j + win_size // 2
                # Get Window
                window = img_resize[center_x - win_size // 2:center_x + win_size // 2 + 1, center_y - win_size // 2:center_y + win_size // 2 + 1, :]
                for channel_index in range(c):
                    # Calculate the histogram of the window
                    hist, _ = np.histogram(window[:, :, channel_index].ravel(), 256, [0, 255])
                    # Calculate the cumulative distribution function
                    cdf = hist.cumsum()
                    # Normalized
                    cdf_normalized = cdf * 255 / cdf[-1]
                    # Put the equalized pixel values back into the original image
                    new_img[i, j, channel_index] = cdf_normalized[img[i, j, channel_index]]
        return new_img
    
    def contrast_limited_ahe(self, img, level = 256, blocks = 8, threshold = 10.0):
        """
        @params img_arr : input image
        @params level : the level of gray scale
        @params window_size : the window used to calculate CDF mapping function
        @params threshold : clip histogram by exceeding the threshold times of the mean value
        @return new_img : output image
        """
        m, n, c = img.shape
        block_m = int(m / blocks)
        block_n = int(n / blocks)
        new_img = np.zeros_like(img, dtype="uint8")
        
        # split small regions and calculate the CDF for each, save to a 2-dim list
        for channel_index in range(c):
            maps = []
            for i in range(blocks):
                row_maps = []
                for j in range(blocks):
                    # block border
                    si, ei = i * block_m, (i + 1) * block_m
                    sj, ej = j * block_n, (j + 1) * block_n
                    
                    # block image array
                    block_img_arr = img[si : ei, sj : ej, channel_index]
                    
                    # calculate histogram and cdf
                    hists, _ = list(np.histogram(block_img_arr.ravel(), 256, [0, 255]))
                    clip_hists = self.clip_histogram_(hists, threshold = threshold)     # clip histogram
                    hists_cdf = (((level - 1) / (block_m * block_n)) * np.cumsum(np.array(clip_hists))).astype("uint8")
                    
                    # save
                    row_maps.append(hists_cdf)
                maps.append(row_maps)
            
            # interpolate every pixel using four nearest mapping functions
            # pay attention to border case
            arr = img[:, :, channel_index].copy()
            for i in range(m):
                for j in range(n):
                    r = int((i - block_m / 2) / block_m)      # the row index of the left-up mapping function
                    c = int((j - block_n / 2) / block_n)      # the col index of the left-up mapping function
                    
                    x1 = (i - (r + 0.5) * block_m) / block_m  # the x-axis distance to the left-up mapping center
                    y1 = (j - (c + 0.5) * block_n) / block_n  # the y-axis distance to the left-up mapping center
                    
                    lu = 0    # mapping value of the left up cdf
                    lb = 0    # left bottom
                    ru = 0    # right up
                    rb = 0    # right bottom
                    
                    # four corners use the nearest mapping directly
                    if r < 0 and c < 0:
                        arr[i][j] = maps[r + 1][c + 1][img[i, j, channel_index]]
                    elif r < 0 and c >= blocks - 1:
                        arr[i][j] = maps[r + 1][c][img[i, j, channel_index]]
                    elif r >= blocks - 1 and c < 0:
                        arr[i][j] = maps[r][c + 1][img[i, j, channel_index]]
                    elif r >= blocks - 1 and c >= blocks - 1:
                        arr[i][j] = maps[r][c][img[i, j, channel_index]]
                    # four border case using the nearest two mapping : linear interpolate
                    elif r < 0 or r >= blocks - 1:
                        if r < 0:
                            r = 0
                        elif r > blocks - 1:
                            r = blocks - 1
                        left = maps[r][c][img[i, j, channel_index]]
                        right = maps[r][c + 1][img[i, j, channel_index]]
                        arr[i][j] = (1 - y1) * left + y1 * right
                    elif c < 0 or c >= blocks - 1:
                        if c < 0:
                            c = 0
                        elif c > blocks - 1:
                            c = blocks - 1
                        up = maps[r][c][img[i, j, channel_index]]
                        bottom = maps[r + 1][c][img[i, j, channel_index]]
                        arr[i][j] = (1 - x1) * up + x1 * bottom
                    # bilinear interpolate for inner pixels
                    else:
                        lu = maps[r][c][img[i, j, channel_index]]
                        lb = maps[r + 1][c][img[i, j, channel_index]]
                        ru = maps[r][c + 1][img[i, j, channel_index]]
                        rb = maps[r + 1][c + 1][img[i, j, channel_index]]
                        arr[i][j] = (1 - y1) * ( (1 - x1) * lu + x1 * lb) + y1 * ( (1 - x1) * ru + x1 * rb)
            arr = arr.astype("uint8")
            new_img[:, :, channel_index] = arr
        return new_img
    
    def bright_wise_histequal(self, img, level = 256):
        """
        @params img : input image
        @params level : gray scale
        @return new_img : output image
        """
        def special_histogram(img_arr, min_v, max_v):
            ### calculate a special histogram with max, min value
            ### @params img_arr : 1-dim numpy.array
            ### @params min_v : min gray scale
            ### @params max_v : max gray scale
            ### @return hists : list type, length = max_v - min_v + 1
            hists = [0 for _ in range(max_v - min_v + 1)]
            for v in img_arr:
                hists[v - min_v] += 1
            return hists
        
        def special_histogram_cdf(hists, min_v, max_v):
            ### calculate a special histogram cdf with max, min value
            ### @params hists : list type
            ### @params min_v : min gray scale
            ### @params max_v : max gray scale
            ### @return hists_cdf : numpy.array
            hists_cumsum = np.cumsum(np.array(hists))
            hists_cdf = (max_v - min_v) / hists_cumsum[-1] * hists_cumsum + min_v
            hists_cdf = hists_cdf.astype("uint8")
            return hists_cdf
        
        def pseudo_variance(arr):
            ### caluculate a type of variance
            ### @params arr : 1-dim numpy.array
            arr_abs = np.abs(arr - np.mean(arr))
            return np.mean(arr_abs)
            
        # search two grayscale level, which can split the image into three parts having approximately same number of pixels
        m, n, c = img.shape
        new_img = np.zeros_like(img, dtype='uint8')
        for channel_index in range(c): 
            hists, _ = list(np.histogram(img[:, :, channel_index].ravel(), 256, [0, 255]))
            hists_arr = np.cumsum(np.array(hists))
            hists_ratio = hists_arr / hists_arr[-1]
            
            scale1 = None
            scale2 = None
            for i in range(len(hists_ratio)):
                if hists_ratio[i] >= 0.333 and scale1 == None:
                    scale1 = i
                if hists_ratio[i] >= 0.667 and scale2 == None:
                    scale2 = i
                    break
            
            # split images
            dark_index = (img[:, :, channel_index] <= scale1)
            mid_index = (img[:, :, channel_index] > scale1) & (img[:, :, channel_index] <= scale2)
            bright_index = (img[:, :, channel_index] > scale2)

            # build three level images
            dark_img_arr = np.zeros_like(img[:, :, channel_index])
            mid_img_arr = np.zeros_like(img[:, :, channel_index])
            bright_img_arr = np.zeros_like(img[:, :, channel_index])
            
            # histogram equalization individually
            dark_hists = special_histogram(img[:, :, channel_index][dark_index], 0, scale1)
            dark_cdf = special_histogram_cdf(dark_hists, 0, scale1)
            
            mid_hists = special_histogram(img[:, :, channel_index][mid_index], scale1, scale2)
            mid_cdf = special_histogram_cdf(mid_hists, scale1, scale2)
            
            bright_hists = special_histogram(img[:, :, channel_index][bright_index], scale2, level - 1)
            bright_cdf = special_histogram_cdf(bright_hists, scale2, level - 1)
            
            # mapping
            dark_img_arr[dark_index] = dark_cdf[img[:, :, channel_index][dark_index]]
            mid_img_arr[mid_index] = mid_cdf[img[:, :, channel_index][mid_index] - scale1]
            bright_img_arr[bright_index] = bright_cdf[img[:, :, channel_index][bright_index] - scale2]
            
            # weighted sum
            #fractor = dark_variance + mid_variance + bright_variance
            arr = dark_img_arr + mid_img_arr + bright_img_arr
            arr = arr.astype("uint8")
            new_img[:, :, channel_index] = arr
        return new_img

    def clip_histogram_(self, hists, threshold = 10.0):
        """
        @params hists : list type
        @params threshold : the top ratio of hists over mean value
        @return clip_hists : list type
        """
        all_sum = sum(hists)
        threshold_value = all_sum / len(hists) * threshold
        total_extra = sum([h - threshold_value for h in hists if h >= threshold_value])
        mean_extra = total_extra / len(hists)
        
        clip_hists = [0 for _ in hists]
        for i in range(len(hists)):
            if hists[i] >= threshold_value:
                clip_hists[i] = int(threshold_value + mean_extra)
            else:
                clip_hists[i] = int(hists[i] + mean_extra)
        
        return clip_hists


def draw_hist(img):
    h, w, c = img.shape
    hists = []
    for i in range(c):
        hist, _ = list(np.histogram(img[0].ravel(), 256, [0, 255]))
        hists.append(hist)
    plt.figure()
    plt.plot(np.linspace(start = 0, stop = 255, num = 256), hists[0], color="red", label='R')
    plt.plot(np.linspace(start = 0, stop = 255, num = 256), hists[1], color="green", label='G')
    plt.plot(np.linspace(start = 0, stop = 255, num = 256), hists[2], color="blue", label='B')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    img = cv2.imread("tungsten_original.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("sceneview.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    draw_hist(img), draw_hist(img2)

    h = HistogramStatisticsEnhance()
    """
    tmp_img1 = h.adaptive_histogram_equalization(img2, 21)
    tmp_img2 = h.adaptive_histogram_equalization(img2, 41)
    tmp_img3 = h.adaptive_histogram_equalization(img2, 61)
    plt.figure()
    plt.imshow(tmp_img1)
    plt.show()
    plt.figure()
    plt.imshow(tmp_img2)
    plt.show()
    plt.figure()
    plt.imshow(tmp_img3)
    plt.show()
    """
    # Choose one method to enhance image
    new_img, new_img2 = h.histogram_equalization(img), h.histogram_equalization(img2)
    new_img, new_img2 = h.adaptive_histogram_equalization(img), h.adaptive_histogram_equalization(img2)
    new_img, new_img2 = h.contrast_limited_ahe(img), h.contrast_limited_ahe(img2)
    new_img, new_img2 = h.bright_wise_histequal(img), h.bright_wise_histequal(img2)
    draw_hist(new_img), draw_hist(new_img2)

    cv2method_img = np.zeros_like(img, dtype = np.uint8)
    cv2method_img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    cv2method_img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    cv2method_img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    cv2method_img2 = np.zeros_like(img2, dtype = np.uint8)
    cv2method_img2[:, :, 0] = cv2.equalizeHist(img2[:, :, 0])
    cv2method_img2[:, :, 1] = cv2.equalizeHist(img2[:, :, 1])
    cv2method_img2[:, :, 2] = cv2.equalizeHist(img2[:, :, 2])
    plt.figure(2)
    plt.subplot(1,3,1), plt.imshow(img)
    plt.subplot(1,3,2), plt.imshow(cv2method_img)
    plt.subplot(1,3,3), plt.imshow(new_img)
    plt.show()

    plt.figure(3)
    plt.subplot(1,3,1), plt.imshow(img2)
    plt.subplot(1,3,2), plt.imshow(cv2method_img2)
    plt.subplot(1,3,3), plt.imshow(new_img2)
    plt.show()

