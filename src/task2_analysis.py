import os
import cv2
import numpy as np
from functions import mean_filter, convolution
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.interactive(True)

path = '../data/task2/'
dir_list = os.listdir(path)

results = []

for file_name in dir_list:
    img = cv2.imread(path+file_name)
    
    for kernel_size in range(3, 17, 2):
        time_start = time()
        kernel = mean_filter(kernel_size)
        mean = convolution(img, kernel)
        time_end = time()
        results.append(time_end - time_start)
        print('Done with kernel size {} for image {}'.format(kernel_size, file_name))
    
    break

plt.plot(np.arange(3, 17, 2), results)
plt.xlabel('Kernel size')
plt.ylabel('Time (s)')
plt.savefig('../output/task2/plot.png')
plt.show()