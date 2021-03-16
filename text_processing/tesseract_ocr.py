from PIL import Image, ImageOps
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from helpers.image import enhance_image, pad_image

def run(keystroke, enhance=True, pad=True):
    im = keystroke.kisolation_frame
    # enhance the image before performing OCR
    # https://stackoverflow.com/questions/42566319/tesseract-ocr-reading-a-low-resolution-pixelated-font-esp-digits
    # https://stackoverflow.com/questions/9480013/image-processing-to-improve-tesseract-ocr-accuracy
    if enhance:
        im = enhance_image(im)

    im = Image.fromarray(im)

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
