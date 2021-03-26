"""
Credit to:
https://stackoverflow.com/questions/28935983/preprocessing-image-for-tesseract-ocr-with-opencv
https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
"""
import cv2
from PIL import Image, ImageOps
import numpy as np
import imutils
from helpers import constants
from helpers.utils import print_full
import matplotlib.pyplot as plt
from operator import itemgetter

def display(im, title="An Image", wait=True):
    im = np.array(im)
    cv2.imshow(title, im)
    if wait:
        cv2.waitKey(0)

def save_image(im, path):
    im = np.array(im)
    cv2.imwrite(path, im)

def enhance_image(im):
    # resize image
    im = cv2.resize(im, None, fx=constants.RESIZE_FACTOR,
                    fy=constants.RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)

    # pyramid mean shift filtering
    # im = cv2.pyrMeanShiftFiltering(im, 21, 51)

    # display(im)

    # convert image to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # canny edge detection
    # im = auto_canny(im)

    # to display the image
    # display(im)

    # automatic thresholding using Otsu's algorithm
    thres, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # im = convexity_defects(im)
    # display(im, "after convdefects")

    # applying dilation and erosion
    kernel = np.ones((1, 1), np.uint8)
    im = cv2.erode(im, kernel, iterations=4)
    # im = cv2.dilate(im, kernel, iterations=1)
    # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    # applying adaptive blur
    # im = cv2.adaptiveThreshold(cv2.bilateralFilter(im, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # im = cv2.adaptiveThreshold(cv2.medianBlur(im, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # applying normal blur
    im = cv2.threshold(cv2.medianBlur(im, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # im = cv2.threshold(cv2.GaussianBlur(im, (1, 1), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # im = cv2.threshold(cv2.bilateralFilter(im, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return im


def pad_image(image, target_size=100):
    padded_image = ImageOps.expand(image, target_size, 'black')
    return padded_image
