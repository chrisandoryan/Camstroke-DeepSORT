import helpers.utils as utils
from helpers.utils import print_info
from helpers import constants
from helpers.font import calc_font_height, calc_font_width

from dataclass.keystroke import KeystrokePoint

class Camstroke(object):
    def __init__(self, video, fps):
        self.last_cursor_position = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax
        self.recorded_fontsizes = []
        # attributes of Camstroke data objects
        self.detected_cursors = []  # object contains data from cursor detection
        # object contains data from keystroke isolation and OCR conversion
        self.isolation_windows = []
        # object contains timing data for Hidden Markov Model learning
        self.keystroke_points = []
        # store video information
        self.video = video
        self.fps = fps

    def get_avg_fontsize(self):
        return utils.calc_average(self.recorded_fontsizes)

    def to_list(self):
        keystroke_data = []
        for cursor, keystroke, font_size in zip(self.detected_cursors, self.isolation_windows, self.recorded_fontsizes):
            merged = dict()
            merged.update(vars(cursor))
            merged.update(vars(keystroke))
            merged['est_fontsize'] = font_size
            keystroke_data.append(merged)
        return keystroke_data

    def get_kpoints(self):
        return [x.get_timing_data(self.fps) for x in self.keystroke_points]

    def _get_existing_kpoint(self, kunit):
        # TODO: find way to accurately compute similarity based on coordinate
        avg_font = self.get_avg_fontsize()
        font_height = calc_font_height(avg_font)
        # font_width = calc_font_width(avg_font)
        print_info("KeystrokePoint Sensitivity (x, y): {}, {}".format(constants.DETECTION_SENSITIVITY, font_height))
        for kp in self.keystroke_points:
            kpoint_xmin, kpoint_ymin, _, _ = kp.last_detection_coordinates
            x_similar = abs(kpoint_xmin - kunit.xmin) <= constants.DETECTION_SENSITIVITY
            y_similar = abs(kpoint_ymin - kunit.ymin) <= constants.DETECTION_SENSITIVITY
            if all((x_similar, y_similar)):
                return kp
        return None

    def store_kunit(self, frame_id, kunit):
        existing_kpoint = self._get_existing_kpoint(kunit)
        if existing_kpoint != None:
            print_info("Storing KUnit to Last KeystrokePoint with ID: %s" % existing_kpoint.id)
            # print("Kunits: ", len(existing_kpoint.kunits))
            existing_kpoint.add_keystroke_unit(frame_id, kunit.get_coordinates(), kunit)
            return existing_kpoint
        else:
            kpoint = KeystrokePoint(frame_id, kunit.get_coordinates())
            kpoint.add_keystroke_unit(frame_id, kunit.get_coordinates(), kunit)
            print_info("Creating New KeystrokePoint with ID: %s" % kpoint.id)
            # print("Kunits: ", len(kpoint.kunits))
            self.keystroke_points.append(kpoint)
            return kpoint