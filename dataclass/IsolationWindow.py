from helpers import constants
from PIL import Image
import numpy as np

class IsolationWindow(object):
    frame_id = -1
    kisolation_frame = None
    kisolation_w = 0
    kisolation_h = 0
    kisolation_xmin = 0
    kisolation_xmax = 0
    kisolation_ymin = 0
    kisolation_ymax = 0
    ocr_result = None

    def __init__(self, frame_id, kisolation_frame, kisolation_coordinates, kisolation_shape):
        self.frame_id = frame_id
        self.kisolation_frame = kisolation_frame
        self.kisolation_w, self.kisolation_h = kisolation_shape
        self.kisolation_xmin, self.kisolation_xmax, self.kisolation_ymin, self.kisolation_ymax = kisolation_coordinates
        return

    def to_image(self):
        return Image.fromarray(self.kisolation_frame)

    def get_character(self):
        index = np.argmax(self.ocr_result['conf'])
        if int(self.ocr_result['conf'][index]) < constants.OCR_CONF_THRESHOLD:
            return None, None
        else:
            text = self.ocr_result['text'][index]
            conf = self.ocr_result['conf'][index]
            if text in constants.INVALID_KEYSTROKE:
                return None, None
            else:
                return conf, text

    def get_isolation_coordinates(self):
        return (self.kisolation_xmin, self.kisolation_ymin, self.kisolation_xmax, self.kisolation_ymax)
