import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities
import scipy


# apply median filter
def applyMedianFilter(img, kSize):
    img = Utilities.ensure_three_channel_grayscale_image(img)
    half_kSize = kSize // 2
    output = np.zeros_like(img)
    #comparison = cv2.medianBlur(img, kSize)
    padded = Utilities.pad_img(img, kSize)

    
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            center_h = h + half_kSize
            center_w = w + half_kSize

            window = padded[
                     center_h - half_kSize: center_h + half_kSize + 1,
                     center_w - half_kSize: center_w + half_kSize + 1
                     ]

            output[h, w] = np.median(window)
            #for c in range(output.shape[2]):
                #output[h, w, c] = np.median(img[h-half_kSize:h+half_kSize+1, w-half_kSize:w+half_kSize+1, c])

    #Comparison with cv2:
    #print(np.allclose(comparison, output))


    return output


# create a moving average kernel of arbitrary size
def createMovingAverageKernel(kSize):
    kernel = np.zeros((kSize, kSize))
    kernel[:,:] = 1/(kSize*kSize)
    return kernel


def gaussian( x, y, sigmaX, sigmaY, meanX, meanY):
    result = 1
    return result


# create a gaussian kernel of arbitrary size
def createGaussianKernel(kSize, sigma=None):
    if sigma is None:
        sigma = 0.3*((kSize-1)*0.5 - 1) + 0.8

    ax = np.arange(-(kSize // 2), kSize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    #normalize:
    kernel /= kernel.sum()

    #kernel_func_1d = cv2.getGaussianKernel(kSize, sigma)
    #kernel_func = np.outer(kernel_func_1d, kernel_func_1d)
    #print(np.allclose(kernel, kernel_func))
    return kernel


# create a sobel kernel in x direction of size 3x3
def createSobelXKernel():
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)
    return sobel_x

# create a sobel kernel in y direction of size 3x3
def createSobelYKernel():
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=float)
    return sobel_y


def applyKernelInSpatialDomain(img, kernel):
    grayscale_img = Utilities.ensure_three_channel_grayscale_image(img)
    filtered_img = scipy.signal.convolve2d(grayscale_img, kernel, mode='same', boundary='symm')
    #comparison = cv2.blur(grayscale_img, (3, 3))
    #print(np.allclose(comparison, np.round(filtered_img).astype(np.uint8)))

    return np.round(filtered_img).astype(np.uint8)


# Extra: create an integral image of the given image
def createIntegralImage(img):
    integral_image = img.copy()
    return integral_image


# Extra: apply the moving average filter by using an integral image
def applyMovingAverageFilterWithIntegralImage(img, kSize):
    filtered_img = img.copy()
    return filtered_img


# Extra:
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    filtered_img = img.copy()
    return filtered_img

def run_runtime_evaluation(img):
    pass