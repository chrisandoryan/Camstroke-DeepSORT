import uuid
from collections import defaultdict
import operator
import numpy as np
from helpers import constants

class KUnit(object):
    def __init__(self, frame_id, kunit_image, kunit_coordinates, kunit_shape):
        self.frame_id = frame_id
        self.kunit_image = kunit_image
        self.width, self.height = kunit_shape
        self.xmin, self.xmax, self.ymin, self.ymax = kunit_coordinates
        self.ocr_result = None

    def set_ocr_result(self, ocr_result):
        self.ocr_result = ocr_result

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
    
    def get_coordinates(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

# KeystrokePoint class contains one or more KUnit, and is used to train HMM for search-space reduction
class KeystrokePoint(object):
    def __init__(self, frame_id, last_detection_coordinates):
        self.id = uuid.uuid4()
        # when the keystroke first appears on the frames, a substitution for KeyPress timing
        self.k_appear = frame_id
        # when the keystroke last appears on the frames, a substitution for KeyRelease timing
        self.k_vanish = 0
        self.last_detection_coordinates = last_detection_coordinates
        self.kunits = []

    def get_keytext(self):
        keytexts = defaultdict(list)

        for k in self.kunits:
            conf, text = k.get_character()
            keytexts[text].append(conf)

        # print(keytexts)
        text, confidences = max(keytexts.items(), key=operator.itemgetter(1))
        print(text != None, confidences != None)

        if text != None and confidences != None:
            return text, (sum(confidences)/len(confidences))
        else:
            return None, None

    def add_keystroke_unit(self, frame_id, last_coordinates, kunit):
        self.k_vanish = frame_id
        self.last_coordinates = last_coordinates
        self.kunits.append(kunit)

    def get_timing_data(self):
        keypress = self.k_appear
        keyrelease = self.k_vanish
        keyhold = (self.k_vanish - self.k_appear) / 2
        keydelay = (self.k_vanish - self.k_appear)

        keytext, confidence = self.get_keytext()

        return {
            'id': self.id,
            'kunits': self.kunits,
            'keypress': keypress,
            'keyrelease': keyrelease,
            'keyhold': keyhold,
            'keydelay': keydelay,
            'keytext': keytext,
            'ocr_conf': confidence
        }


# Old
# KeystrokePoint class contains one or more KUnit, and is used to train HMM for search-space reduction
# class KeystrokePoint(object):
#     def __init__(self, frame_id, last_detection_coordinates):
#         self.id = uuid.uuid4()
#         # when the keystroke first appears on the frames, a substitution for KeyPress timing
#         self.k_appear = frame_id
#         # when the keystroke last appears on the frames, a substitution for KeyRelease timing
#         self.k_vanish = 0
#         self.last_detection_coordinates = last_detection_coordinates
#         self.kunits = []

#     def get_keytext(self):
#         keytexts = defaultdict(list)

#         for k in self.kunits:
#             conf, text = k.get_character()
#             keytexts[text].append(conf)

#         # print(keytexts)
#         text, confidences = max(keytexts.items(), key=operator.itemgetter(1))

#         return text, (sum(confidences)/len(confidences))

#     def add_keystroke_unit(self, frame_id, last_detection_coordinates, isolation_window):
#         self.k_vanish = frame_id
#         self.last_detection_coordinates = last_detection_coordinates
#         self.kunits.append(isolation_window)

#     def get_timing_data(self):
#         keypress = self.k_appear
#         keyrelease = self.k_vanish
#         keyhold = (self.k_vanish - self.k_appear) / 2
#         keydelay = (self.k_vanish - self.k_appear)

#         keytext, confidence = self.get_keytext()

#         return {
#             'id': self.id,
#             'kunits': self.kunits,
#             'keypress': keypress,
#             'keyrelease': keyrelease,
#             'keyhold': keyhold,
#             'keydelay': keydelay,
#             'keytext': keytext,
#             'ocr_conf': confidence
#         }
