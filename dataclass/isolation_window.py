from helpers import constants
from PIL import Image
import numpy as np

class IsolationWindow(object):
    def __init__(self, frame_id, kisolation_frame, kisolation_coordinates, kisolation_shape):
        self.frame_id = frame_id
        self.kisolation_frame = kisolation_frame
        self.kisolation_w, self.kisolation_h = kisolation_shape
        self.kisolation_xmin, self.kisolation_xmax, self.kisolation_ymin, self.kisolation_ymax = kisolation_coordinates
        return

    def to_image(self):
        return Image.fromarray(self.kisolation_frame)

    def get_isolation_coordinates(self):
        return (self.kisolation_xmin, self.kisolation_ymin, self.kisolation_xmax, self.kisolation_ymax)
