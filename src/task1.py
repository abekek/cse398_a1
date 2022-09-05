import os
import cv2
from functions import mean_filter, convolution

path = '../data/task1/'
dir_list = os.listdir(path)

for file_name in dir_list:
    img = cv2.imread(path+file_name)
    cv2.imshow('input_image', img)
    cv2.waitKey(0)
    
    kernel_size = 15
    kernel = mean_filter(kernel_size)
    
    blurred = convolution(img, kernel)
    cv2.imshow('blurred_image', blurred)
    cv2.waitKey(0)

    residual = img/255 - blurred
    output = img/255 + residual
    
    cv2.imshow('output_image', output)
    cv2.imwrite('../output/task1/{}.jpg'.format(file_name[0]), output * 255)
    cv2.waitKey(0)

    cv2.destroyAllWindows()