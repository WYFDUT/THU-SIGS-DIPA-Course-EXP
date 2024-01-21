import cv2
import numpy as np
import degraded
import reconstruct
import matplotlib.pyplot as plt


if __name__ == "__main__":
    img_path = "C:\\Users\\WYF\\Desktop\\DIP\\Image_Degradations_and_Reconstruction\\DIP.bmp"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = np.float64(image)/255
    img_dft = np.fft.fftshift(np.fft.fft2(img))

    #单独湍流模糊退化
    H_turbulenceBlur = degraded.turbulenceBlur(image, k=0.001)
    dst_fft_shift = img_dft * H_turbulenceBlur
    dst = np.fft.ifft2(np.fft.ifftshift(dst_fft_shift))
    dst = np.abs(dst)*255
    dst_turbulence = np.uint8(cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("Turbulence Blur", dst_turbulence)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #单独运动模糊退化
    img_dft = np.fft.fft2(img)
    H_PSF = degraded.getMotionDsf(image.shape, 30, 1e-6, a=-0.1, b=0.1)
    dst_fft_shift = img_dft * H_PSF
    dst = np.fft.ifft2(dst_fft_shift)
    dst = np.abs(np.fft.fftshift(dst))
    dst_motion = np.uint8(cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("Motion Blur", dst_motion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #叠加高斯噪声
    noisy_turbulence = dst_turbulence.std() * np.random.normal(loc=0.0, scale=0.05, size=dst_turbulence.shape)
    img_turbulence_noisy = np.uint8(cv2.normalize(noisy_turbulence + dst_turbulence, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("Noisy Turbulence Blur", img_turbulence_noisy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    noisy_motion = dst_motion.std() * np.random.normal(loc=0.0, scale=0.05, size=dst_motion.shape)
    img_motion_noisy = np.uint8(cv2.normalize(noisy_motion + dst_motion, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("Noisy Motion Blur", img_motion_noisy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #最佳陷波滤波
    img_turbulence_Notch = reconstruct.optimumNotchFilter(img_turbulence_noisy, noisy_turbulence)
    img_turbulencen_Notch = np.uint8(cv2.normalize(img_turbulence_Notch, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("test", np.uint8(img_turbulence_Notch))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_motion_Notch = reconstruct.optimumNotchFilter(img_motion_noisy, noisy_motion)
    img_motion_Notch = np.uint8(cv2.normalize(img_motion_Notch, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("test", np.uint8(img_motion_Notch))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    #维纳滤波
    img_motion_Wiener = reconstruct.wienerFilter(img_motion_noisy, H_PSF)
    cv2.imshow("test", np.uint8(cv2.normalize(img_motion_Wiener, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_turbulence_Wiener = reconstruct.wienerFilter2(img_turbulence_noisy, H_turbulenceBlur)
    cv2.imshow("test", np.uint8(cv2.normalize(img_turbulence_Wiener, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


