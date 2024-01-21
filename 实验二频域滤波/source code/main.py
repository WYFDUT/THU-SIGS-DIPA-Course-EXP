import cv2
import numpy as np

import DFT
import Lowpass_Filter
import matplotlib.pyplot as plt


if __name__ == "__main__":
    lena = cv2.cvtColor(cv2.imread("C:\\Users\\WYF\\Desktop\\DIP\\Filtering_in_the_Frequency_Domain\\Lena.png"), cv2.COLOR_BGR2GRAY)
    sample = cv2.cvtColor(cv2.imread("C:\\Users\\WYF\\Desktop\\DIP\\Filtering_in_the_Frequency_Domain\\Sample.png"), cv2.COLOR_BGR2GRAY)

    lena = lena/255
    sample = sample/255

    dft = DFT.DiscreteFourierTrans()
    lp_filter = Lowpass_Filter.LowpassFilter()

    '''
    # DFT
    lena_DFT =  dft.fourier_transform_2d(lena)
    sample_DFT =  dft.fourier_transform_2d(sample)

    lena_DFT_img = np.array(np.log10(1 + abs(lena_DFT)))
    sample_DFT_img = np.array(np.log10(1 + abs(sample_DFT)))
    
    plt.figure(1)
    plt.imshow(lena_DFT_img,cmap="gray")
    plt.show()
    plt.figure(2)
    plt.imshow(sample_DFT_img,cmap="gray")
    plt.show()
    '''

    '''
    #IDFT
    lena_IDFT =  dft.inverse_fourier_transform_2d(lena_DFT)
    sample_IDFT =  dft.inverse_fourier_transform_2d(sample_DFT)

    lena_IDFT_img = np.array(abs(lena_IDFT))
    sample_IDFT_img = np.array(abs(sample_IDFT))
    
    plt.figure(3)
    plt.imshow(lena_IDFT_img,cmap="gray")
    plt.show()
    plt.figure(4)
    plt.imshow(sample_IDFT_img,cmap="gray")
    plt.show()
    '''

    '''
    #Filter
    test3 = lp_filter.ideal_lowpass_filter(lena_DFT, 40)
    test4 = lp_filter.ideal_lowpass_filter(sample_DFT, 40)
    #test3 = lp_filter.gaussian_lowpass_filter(lena_DFT, 40)
    #test4 = lp_filter.gaussian_lowpass_filter(sample_DFT, 40)
    test3 =  dft.inverse_fourier_transform_2d(test3)
    test4 =  dft.inverse_fourier_transform_2d(test4)

    test3 = np.array(abs(test3))
    test4 = np.array(abs(test4))
    plt.figure(5)
    plt.imshow(test3,cmap="gray")
    plt.show()
    plt.figure(6)
    plt.imshow(test4,cmap="gray")
    plt.show()
    '''

    


