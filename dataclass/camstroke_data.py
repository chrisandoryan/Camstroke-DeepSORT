import helpers.utils as utils
from helpers import constants

from dataclass.keystroke import KeystrokePoint

class Camstroke(object):
    def __init__(self):
        self.last_cursor_position = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax
        self.recorded_fontsizes = []
        # attributes of Camstroke data objects
        self.detected_cursors = []  # object contains data from cursor detection
        # object contains data from keystroke isolation and OCR conversion
        self.isolation_windows = []
        # object contains timing data for Hidden Markov Model learning
        self.keystroke_points = []

    def get_avg_fontsize(self):
        return utils.calc_average(self.recorded_fontsizes)

    def get_all_data(self):
        keystroke_data = []
        for cursor, keystroke, font_size in zip(self.detected_cursors, self.isolation_windows, self.recorded_fontsizes):
            merged = dict()
            merged.update(vars(cursor))
            merged.update(vars(keystroke))
            merged['est_fontsize'] = font_size
            keystroke_data.append(merged)
        return keystroke_data

    def store_kunit(self, kunit):
        if len(self.keystroke_points) > 0:
            last_kpoint = self.keystroke_points[-1]
            last_xmin, _, _, _ = last_kpoint.last_detection_coordinates

        return

    def merge_keystroke_to_keystroke_points(self, frame_id, isolation_window):
        if len(self.keystroke_points) > 0:
            last_kpoint = self.keystroke_points[-1]
            last_xmin, _, _, _ = last_kpoint.last_detection_coordinates
            if abs(isolation_window.kisolation_xmin - last_xmin) <= constants.DETECTION_SENSITIVITY:
                # print("Using Last KeystrokePoint with ID: ", last_kpoint.id)
                # print("Kunits: ", len(last_kpoint.kunits))
                last_kpoint.add_keystroke_unit(
                    frame_id, isolation_window.get_isolation_coordinates(), isolation_window)
                return last_kpoint

        kpoint = KeystrokePoint(
            frame_id, isolation_window.get_isolation_coordinates())
        # print("Creating New KeystrokePoint with ID: ", kpoint.id)
        # print("Kunits: ", len(kpoint.kunits))
        kpoint.add_keystroke_unit(
            frame_id, isolation_window.get_isolation_coordinates(), isolation_window)
        self.keystroke_points.append(kpoint)
        return kpoint