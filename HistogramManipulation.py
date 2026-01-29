import cv2
import numpy as np
import Utilities




# Task 1
# function to stretch an image
def stretchHistogram(img):
    img_float = img.astype(np.float32)
    if img.ndim == 2:
        stretched = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX)
        return stretched.astype(np.uint8)

    channels = cv2.split(img_float)
    stretched_channels = [cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX) for ch in channels]
    return cv2.merge(stretched_channels).astype(np.uint8)

# Task 2
# function to equalize an image
def equalizeHistogram(img):
    if img.ndim == 2:
        return cv2.equalizeHist(img)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

#Hilfsfunktion
# function to apply a look-up table onto an image
def applyLUT(img, LUT):
    lut = np.asarray(LUT, dtype=np.uint8)
    if lut.size != 256:
        raise ValueError("LUT must contain 256 entries")
    return cv2.LUT(img, lut)

# Hilfsfunktion
# function to find the minimum an maximum in a histogram
def findMinMaxPos(histogram):
    histogram = np.asarray(histogram).flatten()
    non_zero_indices = np.nonzero(histogram)[0]
    if non_zero_indices.size == 0:
        return 0, 0
    return int(non_zero_indices[0]), int(non_zero_indices[-1])

# Hilfsfunktion
# function to create a vector containing the histogram
def calculateHistogram(img, nrBins):
    # create histogram vector
    gray = Utilities.ensure_one_channel_grayscale_image(img)
    histogram = cv2.calcHist([gray], [0], None, [nrBins], [0, 256]).flatten().astype(int)
    return histogram

def apply_log(img):
    img_float = img.astype(np.float32)
    max_val = img_float.max()
    if max_val == 0:
        return np.zeros_like(img)

    c = 255.0 / np.log1p(max_val)
    result = c * np.log1p(img_float)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_exp(img):
    img_norm = img.astype(np.float32) / 255.0
    exp_img = np.expm1(img_norm)
    max_val = exp_img.max()
    if max_val == 0:
        return np.zeros_like(img)

    result = (exp_img / max_val) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_inverse(img):
    return 255 - img

def apply_threshold(img, threshold):
    gray = Utilities.ensure_one_channel_grayscale_image(img)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
