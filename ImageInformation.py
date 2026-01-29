import cv2
import numpy as np

# Example for basic pixel based image manipulation:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html

# Task 1:   Implement some kind of noticeable image manipulation in this function
#           e.g. channel manipulation, filter you already know, drawings on the image etc.
def myFirstImageManipulation(img):
    if img.ndim == 2:  # normalize grayscale to three channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 0.8, edges_bgr, 0.8, 0)  # overlay edges for a comic look
    return result



# Task 2:   Return the basic image properties to the console:
#           width, height,
#           the color of the first pixel of the image,
#           Color of the first pixel in the second row
#           Color of the first pixel in the second column
#           This function should work for images with three channels

def imageSize(img):
    if img.ndim < 2:
        raise ValueError("Invalid image: missing spatial dimensions")

    height, width = img.shape[:2]
    print(f"Width: {width}, Height: {height}")
    return [width, height]

def getPixelColor(img):
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Function expects a three-channel image")

    height, width, _ = img.shape
    if height < 2 or width < 2:
        raise ValueError("Image too small to read requested pixels")

    first_pixel = img[0, 0].tolist()
    first_pixel_second_row = img[1, 0].tolist()
    first_pixel_second_column = img[0, 1].tolist()

    print(f"First pixel (0,0): {first_pixel}")
    print(f"First pixel in second row (1,0): {first_pixel_second_row}")
    print(f"First pixel in second column (0,1): {first_pixel_second_column}")
    return [first_pixel, first_pixel_second_row, first_pixel_second_column]

# Task 3:   Separate the given channels of a colour image in this function and return it as separate image
#           the separate image need three channels
#
def returnChannel(img, channel):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0 (B), 1 (G), or 2 (R)")

    result = np.zeros_like(img)
    result[:, :, channel] = img[:, :, channel]
    return result
