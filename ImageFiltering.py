import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import Utilities
import scipy
import time

def pad_img(img, kSize_y, kSize_x=None, pad_mode="edge"):
    if kSize_x is None:
        kSize_x = kSize_y

    pad_y = kSize_y // 2
    pad_x = kSize_x // 2

    if len(img.shape) == 2:  # grayscale
        padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode=pad_mode)
    else:  # color image (H x W x C)
        padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode=pad_mode)

    return padded

# apply median filter
def applyMedianFilter(img, kSize):
    img = Utilities.ensure_three_channel_grayscale_image(img)
    half_kSize = kSize // 2
    output = np.zeros_like(img)
    #comparison = cv2.medianBlur(img, kSize)
    padded = pad_img(img, kSize)

    
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
        sigma = kSize/2 #laut Aufgabenstellung S.7 Schöner in OpenCV: 0.3*((kSize-1)*0.5 - 1) + 0.8

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


def applyKernelInSpatialDomain(img, kernel, round=True):
    grayscale_img = Utilities.ensure_one_channel_grayscale_image(img)

    k_size_h, k_size_w = kernel.shape
    padded = pad_img(grayscale_img, k_size_h, k_size_w)

    #fertige funktion:
    #filtered_img_comp = scipy.signal.convolve2d(grayscale_img, kernel, mode='same', boundary='symm')

    #selbst implementiert:
    h, w = grayscale_img.shape
    filtered_img = np.zeros((h, w))
    #flip kernel
    kernel = np.flipud(np.fliplr(kernel))

    # ---- vertical 1D kernel ----
    if k_size_w == 1:
        for i in range(h):
            for j in range(w):
                column = padded[i:i + k_size_h, j]
                filtered_img[i, j] = np.sum(column * kernel[:, 0])

    # ---- horizontal 1D kernel ----
    elif k_size_h == 1:
        for i in range(h):
            for j in range(w):
                row = padded[i, j:j + k_size_w]
                filtered_img[i, j] = np.sum(row * kernel[0, :])

    # ---- general 2D kernel ----
    else:
        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_size_h, j:j + k_size_w]
                filtered_img[i, j] = np.sum(region * kernel)
    #comparison = cv2.blur(grayscale_img, (k_size_h, k_size_w))
    #print(np.allclose(comparison, np.round(filtered_img).astype(np.uint8)))

    if round:
        return np.round(filtered_img).astype(np.uint8)
    else:
        return filtered_img

def applyMovingAverageFilterWithConv(img, kSize):
    kernel = createMovingAverageKernel(kSize)
    filtered_img = applyKernelInSpatialDomain(img, kernel)
    return Utilities.ensure_three_channel_grayscale_image(filtered_img)

# Extra: create an integral image of the given image
def createIntegralImage(img):
    h, w = img.shape
    integral = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            val = img[y, x]

            left = integral[y, x - 1] if x > 0 else 0
            top = integral[y - 1, x] if y > 0 else 0
            top_left = integral[y - 1, x - 1] if (x > 0 and y > 0) else 0

            integral[y, x] = val + left + top - top_left

    return integral


def rect_sum(integral, x1, y1, x2, y2):

    A = integral[y1-1, x1-1] if x1 > 0 and y1 > 0 else 0
    B = integral[y1-1, x2] if y1 > 0 else 0
    C = integral[y2, x1-1] if x1 > 0 else 0
    D = integral[y2, x2]

    return D - B - C + A


# Extra: apply the moving average filter by using an integral image
def applyMovingAverageFilterWithIntegralImage(img, kSize):
    grayscale_img = Utilities.ensure_one_channel_grayscale_image(img)
    h, w = grayscale_img.shape

    padded = pad_img(grayscale_img, kSize)
    integral = createIntegralImage(padded)

    filtered_img = np.zeros(grayscale_img.shape)

    for y in range(h):
        for x in range(w):
            y1 = y
            x1 = x
            y2 = y + kSize - 1
            x2 = x + kSize - 1

            sum = rect_sum(integral, x1, y1, x2, y2)

            #sum divided by sample amount = average
            filtered_img[y, x] = sum / (kSize*kSize)

    #kernel = createMovingAverageKernel(kSize)
    #comp = applyKernelInSpatialDomain(img, kernel)
    #diff = comp - filtered_img
    #print(f"all close: {np.allclose(comp, np.round(filtered_img).astype(np.uint8))}")

    return np.round(filtered_img).astype(np.uint8)


# Extra:
def applyMovingAverageFilterWithSeperatedKernels(img, kSize):
    # unnecessary since MovingAverage kernels are always symmetrical:
    # if np.linalg.det(kernel) == 0:
    # seperated 1d kernel for a moving average filter is just:
    kernel = 1/kSize*np.ones((kSize, 1))

    intermediate = applyKernelInSpatialDomain(img, kernel, round=False)
    filtered_img = applyKernelInSpatialDomain(intermediate, kernel.T)

    #kernel = createMovingAverageKernel(kSize)
    #comp = applyKernelInSpatialDomain(img, kernel)
    #diff = comp - filtered_img
    #print(f"all close: {np.allclose(comp, filtered_img)}")

    return filtered_img

def run_runtime_evaluation(img):
    #overhead from slicing the array makes applyMovingAverageFilterWithSeperatedKernels slower than the normal conv for small kernels. Use bigger kernel sizes to actually see speed boost
    functions = [applyMovingAverageFilterWithConv, applyMovingAverageFilterWithSeperatedKernels, applyMovingAverageFilterWithIntegralImage]

    for w in range(3, 16, 2):  # 3,5,7,...,15
        #w = 201
        for function in functions:
            start = time.perf_counter()

            result = function(img, w)

            end = time.perf_counter()
            runtime = end - start

            print(f"Kernelgröße {w}x{w} für Funktion {function.__name__}: {runtime:.6f} Sekunden")