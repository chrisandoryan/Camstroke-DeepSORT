"""
Credit to:
https://stackoverflow.com/questions/28935983/preprocessing-image-for-tesseract-ocr-with-opencv
"""
import cv2
from PIL import Image, ImageOps
import numpy as np

RESIZE_FACTOR = 5

def enhance_image(im):
    # print("IM Shape Before: ", im.shape)
    # resize image
    im = cv2.resize(im, None, fx=RESIZE_FACTOR,
                    fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)

    # print("IM Shape After: ", im.shape)

    # convert image to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # automatic thresholding using Otsu's algorithm
    thres, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # applying dilation and erosion
    kernel = np.ones((1, 1), np.uint8)
    im = cv2.dilate(im, kernel, iterations=6)
    im = cv2.erode(im, kernel, iterations=6)

    # applying adaptive blur
    # im = cv2.adaptiveThreshold(cv2.bilateralFilter(im, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # im = cv2.adaptiveThreshold(cv2.medianBlur(im, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # applying normal blur
    im = cv2.threshold(cv2.medianBlur(im, 3), 0, 255,
                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # im = cv2.threshold(cv2.bilateralFilter(im, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # im = cv2.threshold(cv2.GaussianBlur(im, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return im


def pad_image(image, target_size=100):
    padded_image = ImageOps.expand(image, target_size, 'black')
    return padded_image
