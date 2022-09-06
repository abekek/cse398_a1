import os
import cv2
import numpy as np
from functions import convolution_sobel, sobel_horizontal_filter, sobel_vertical_filter

path = '../data/task3/'
dir_list = os.listdir(path)

for file_name in dir_list:
    img = cv2.imread(path+file_name)
    cv2.imshow('input_image', img)
    cv2.waitKey(0)
    
    sobel_hor = sobel_horizontal_filter()
    sobel_ver = sobel_vertical_filter()
    sobel = convolution_sobel(img, sobel_hor, sobel_ver)
    cv2.imshow('sobel', sobel)
    cv2.imwrite('../output/task3/{}'.format(file_name), sobel * 255)
    cv2.waitKey(0)

    cv2.destroyAllWindows()