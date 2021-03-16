from PIL import Image, ImageOps
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from text_processing import cca

def enhance_image(im):
    # resize image
    RESIZE_FACTOR = 4
    im = cv2.resize(im, None, fx=RESIZE_FACTOR,
                    fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)

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


def run(keystroke, font_size, enhance=True, pad=True):
    im = keystroke.kisolation_frame
    # enhance the image before performing OCR
    # https://stackoverflow.com/questions/42566319/tesseract-ocr-reading-a-low-resolution-pixelated-font-esp-digits
    # https://stackoverflow.com/questions/9480013/image-processing-to-improve-tesseract-ocr-accuracy
    if enhance:
        im = enhance_image(im)

    im = Image.fromarray(im)

    # perform connected component labelling
    cca.run_with_stats(im, font_size)

    # invert the image, tesseract works best with black font and white background
    im = ImageOps.invert(im)

    # perform image padding and resize for higher resolution
    if pad:
        im = pad_image(im, target_size=50)

    # basic configuration
    # return im, pytesseract.image_to_string(im, config='--psm 10').strip()

    # if need to limit charset, use this instead:
    # return im, pytesseract.image_to_data(im, output_type=Output.DICT, config='--psm 10 --oem 0 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz')

    # stable configuration
    # print("Shape: ", im.size)
    return im, pytesseract.image_to_data(im, output_type=Output.DICT, config='--psm 10 --oem 3')
