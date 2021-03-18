import uuid
from collections import defaultdict
import operator

class KUnit(object):
    def __init__(self):
        return

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

    def keytext_in_consensus(self):
        keytexts = defaultdict(list)

        for k in self.kunits:
            conf, text = k.get_character()
            keytexts[text].append(conf)

        # print(keytexts)
        text, confidences = max(keytexts.items(), key=operator.itemgetter(1))

        return text, (sum(confidences)/len(confidences))

    def add_keystroke_unit(self, frame_id, last_detection_coordinates, isolation_window):
        self.k_vanish = frame_id
        self.last_detection_coordinates = last_detection_coordinates
        self.kunits.append(isolation_window)

    def get_timing_data(self):
        keypress = self.k_appear
        keyrelease = self.k_vanish
        keyhold = (self.k_vanish - self.k_appear) / 2
        keydelay = (self.k_vanish - self.k_appear)

        keytext, confidence = self.keytext_in_consensus()

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
