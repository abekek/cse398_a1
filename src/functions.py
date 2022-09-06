import numpy as np
import cv2

# this function is used to calculate the gaussian distribution
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

# manual implementation of horizontal sobel filter
def sobel_horizontal_filter():
    kernel = np.array([[-2, -2, -4, -2, -2], [-1, -1, -2, -1, -1], [0, 0, 0, 0, 0], [1, 1, 2, 1, 1], [2, 2, 4, 2, 2]])
    return kernel

# manual implementation of vertical sobel filter
def sobel_vertical_filter():
    kernel = np.array([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])
    return kernel
 
# implementation of median filter
def median_filter(data):
    kernel = data
    kernel_size = kernel.shape[0]
    kernel = np.sort(kernel, axis=None)
    new_kernel = np.reshape(kernel, (kernel_size, kernel_size))
    return new_kernel

# implementation of gaussian filter
def gaussian_filter(kernel_size, sigma=1):
    kernel_1D = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D

# implementation of mean filter
def mean_filter(kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / (kernel_size**2)
    return kernel

# implementation of convolution for the sobel filter
def convolution_sobel(image, horizontal, vertical):
    # get the image and kernel dimensions
    image_row, image_col = image.shape[0], image.shape[1]
    kernel_row, kernel_col = horizontal.shape
    
    # we just need one channel for the sobel filter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = np.zeros(image.shape)

    # pad the image
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    # create a padded image
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    # copy the image to the padded image
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    # apply the sobel filter
    for row in range(image_row):
        for col in range(image_col):
            f_x = np.sum(padded_image[row:row + kernel_row, col:col + kernel_col] * horizontal)
            f_y = np.sum(padded_image[row:row + kernel_row, col:col + kernel_col] * vertical)
            magnitude = np.sqrt(f_x**2 + f_y**2)
            phase = np.arctan(f_y / f_x)
            output[row, col] = magnitude
            # print('Magnitude at row: {}, col: {} is {}'.format(row, col, magnitude))
            # print('Phase at row: {}, col: {} is {}'.format(row, col, phase))
            output[row, col] /= (kernel_row**2 * kernel_col**2) # normalize the output
    
    return output

# implementation of generic convolution
def convolution(image, kernel, useMedianKernel=False):
    # get the image and kernel dimensions
    image_row, image_col = image.shape[0], image.shape[1]
    kernel_row, kernel_col = kernel.shape
    
    # stacking the kernel to get the 3 channels
    output = np.zeros(image.shape)
    kernel = np.dstack([kernel]*3)
 
    # pad the image
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    # create a padded image
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width), output.shape[2]))
 
    # copy the image to the padded image
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width, :] = image

    # apply the kernel using convolution
    for row in range(image_row):
        for col in range(image_col):
            for channel in range(output.shape[2]):
                # if we want to use the median filter
                if useMedianKernel:
                    kernel = median_filter(padded_image[row:row + kernel_row, col:col + kernel_col, channel])
                    kernel = np.dstack([kernel]*3)
                    output[row, col, channel] = kernel[kernel_row // 2, kernel_col // 2, channel]
                # else we use the generic convolution
                else:
                    output[row, col, channel] = np.sum(padded_image[row:row + kernel_row, col:col + kernel_col, channel] * kernel[:, :, channel])
            output[row, col, :] /= kernel_row * kernel_col # normalize the output
 
    return output
