import cv2
import numpy as np
import Utilities

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

#Anzahl möglicher Grauwerte
L = 256

def applyLUT(img, lut):#
    """
    Conceptually:

    result = img.copy()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for intensity in range(img.shape[2]):
                result[x,y,intensity] = lut[img[x,y,intensity]]

    But faster:
    """
    
    return lut[img].astype(np.uint8)


def equalizeHistogram(img):
    result = img.copy()
    gray = Utilities.ensure_one_channel_grayscale_image(result)
    histogram = calculateHistogram(gray, L)

    lut = (np.round(((L-1)/gray.size)*(np.cumsum(histogram)))).astype(np.uint8)
    result = applyLUT(gray, lut)
    #For comparison:
    #comp = cv2.equalizeHist(gray)
    #print(np.allclose(comp, result))
    result = Utilities.ensure_three_channel_grayscale_image(result)

    return result, lut

def findMinMaxPos(histogram):
    histogram = np.asarray(histogram).flatten()
    non_zero_indices = np.nonzero(histogram)[0]
    if non_zero_indices.size == 0:
        return 0, 0
    return int(non_zero_indices[0]), int(non_zero_indices[-1])

def stretchHistogram(img):
    result = img.copy()
    histogram = calculateHistogram(img, L)

    #0 = ax_min + n and 256 = ax_max + n
    solution = np.array([0, L])
    x_min, x_max = findMinMaxPos(histogram)
    equation = np.array([[x_min, 1], [x_max, 1]])
    x = np.linalg.solve(equation, solution)

    lut = np.clip(Utilities.create_identity_lut()*x[0] + x[1], a_min=0, a_max=L-1).astype(np.uint8)

    result = applyLUT(result, lut)

    return result, lut

def calculateHistogram(img, nrBins):
    # create histogram vector
    gray = Utilities.ensure_one_channel_grayscale_image(img)
    histogram = cv2.calcHist([gray], [0], None, [nrBins], [0, 256]).flatten().astype(int)
    return histogram

def apply_log(img):
    lut = Utilities.create_identity_lut()


    return applyLUT(img, lut), lut

def apply_exp(img):
    lut = Utilities.create_identity_lut()


    return applyLUT(img, lut), lut

def apply_inverse(img):
    #works too but doesnt produce a lut:
    #comp = (L-1)-img
    lut = np.invert(Utilities.create_identity_lut().astype(np.uint8))
    result = applyLUT(img, lut)

    return result, lut

def apply_threshold(img, threshold):
    lut = Utilities.create_identity_lut()
    lut[threshold:] = 255
    lut[:threshold] = 0

    return applyLUT(img, lut), lut


def apply_contrast_sigmoid(img, factor):
    lut =  np.linspace(0, 1, 256)

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return applyLUT(img, lut), lut

def apply_contrast(img, factor):
    lut = Utilities.create_identity_lut()
    lut = np.clip(lut*factor, 0 , 255).astype(np.uint8)
    return applyLUT(img, lut), lut


def apply_exposure(img, ev):
    lut = Utilities.create_identity_lut()

    return applyLUT(img, lut), lut