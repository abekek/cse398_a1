import os
import cv2
import numpy as np
from functions import gaussian_filter, mean_filter, convolution

path = '../data/task2/'
dir_list = os.listdir(path)

for file_name in dir_list:
    img = cv2.imread(path+file_name)
    cv2.imshow('input_image', img)
    cv2.waitKey(0)
    
    kernel_size = 15
    kernel = mean_filter(kernel_size)
    mean = convolution(img, kernel)
    cv2.imshow('mean_image', mean)
    cv2.imwrite('../output/task2/mean/{}'.format(file_name), mean * 255)
    cv2.waitKey(0)

    kernel_size = 45
    kernel = gaussian_filter(kernel_size)
    gauss = convolution(img, kernel)
    cv2.imshow('gauss_image', gauss)
    cv2.imwrite('../output/task2/gauss/{}'.format(file_name), gauss * 255)
    cv2.waitKey(0)

    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size))
    median = convolution(img, kernel, useMedianKernel=True)
    cv2.imshow('median_image', median)
    cv2.imwrite('../output/task2/median/{}'.format(file_name), median * 255)
    cv2.waitKey(0)

    cv2.destroyAllWindows()