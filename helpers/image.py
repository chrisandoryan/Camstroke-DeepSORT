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
    # display(im, "Resizing")

    # convert image to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # display(im, "Grayscale")
    
    # canny edge detection
    # im = auto_canny(im)

    # display(im, "Before Thresholding")
    # automatic thresholding using Otsu's algorithm
    thres, im = cv2.threshold(im, 215, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # display(im, "After Thresholding")

    # applying dilation and erosion
    kernel = np.ones((3, 1), np.uint8)
    im = cv2.erode(im, kernel, iterations=6)
    im = cv2.dilate(im, kernel, iterations=1)

    kernel = np.ones((1, 3), np.uint8)
    im = cv2.erode(im, kernel, iterations=6)
    im = cv2.dilate(im, kernel, iterations=1)
    # display(im, "Erode Dilate")

    # blurring and dilating to restore boldness
    # NOTE: this will make the font bolder
    im = cv2.GaussianBlur(im, (11, 11), 0)
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.dilate(im, kernel, iterations=3)
    # display(im, "Blurring and Dilating")

    # applying adaptive blur
    # im = cv2.adaptiveThreshold(cv2.bilateralFilter(im, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # im = cv2.adaptiveThreshold(cv2.medianBlur(im, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # applying normal blur
    # im = cv2.threshold(cv2.medianBlur(im, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    im = cv2.threshold(cv2.GaussianBlur(im, (1, 1), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # im = cv2.threshold(cv2.bilateralFilter(im, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # display(im, "Final")

    return im


def pad_image(image, target_size=100):
    padded_image = ImageOps.expand(image, target_size, 'black')
    return padded_image
