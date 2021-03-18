class DetectedCursor(object):
    detection_id = ""
    frame_id = ""
    score = 0
    bbox_xmin = 0
    bbox_ymin = 0
    bbox_xmax = 0
    bbox_ymax = 0
    bbox_w = 0
    bbox_h = 0

    def __init__(self, detection_id, frame_id, score, detection_coordinates, detection_shape):
        self.detection_id = detection_id
        self.frame_id = frame_id
        self.score = score
        self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax = detection_coordinates
        self.bbox_w, self.bbox_h = detection_shape
        return
