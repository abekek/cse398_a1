import numpy as np

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
def median_filter(kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    pass

def gaussian_filter(kernel_size, sigma=1):
    kernel_1D = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D

def mean_filter(kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / (kernel_size**2)
    return kernel

def convolution(image, kernel):
    image_row, image_col = image.shape[0], image.shape[1]
    kernel_row, kernel_col = kernel.shape
    
    output = np.zeros(image.shape)

    kernel = np.dstack([kernel]*3)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width), output.shape[2]))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width, :] = image

    for row in range(image_row):
        for col in range(image_col):
            for channel in range(output.shape[2]):
                output[row, col, channel] = np.sum(padded_image[row:row + kernel_row, col:col + kernel_col, channel] * kernel[:, :, channel])
            output[row, col, :] /= kernel_row * kernel_col
 
    return output
