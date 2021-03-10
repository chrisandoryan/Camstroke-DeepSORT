from yolo_deepsort import cursor_tracker, cursor_detector
from PIL import Image, ImageOps
import cv2
import pytesseract
from math import sqrt, floor
import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import itertools
from pytesseract import Output
import uuid
from collections import defaultdict
import operator
from helpers import constants
import helpers.utils as utils
import argparse


class Camstroke(object):
    def __init__(self):
        self.last_cursor_position = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax
        self.recorded_fontsizes = []
        # attributes of Camstroke data objects
        self.detected_cursors = []  # object contains data from cursor detection
        # object contains data from keystroke isolation and OCR conversion
        self.isolated_keystrokes = []
        # object contains timing data for Hidden Markov Model learning
        self.keystroke_points = []

    def get_avg_fontsize(self):
        return calc_average(self.recorded_fontsizes)

    def get_camstroke_detection_data(self):
        keystroke_data = []
        for cursor, keystroke, font_size in zip(self.detected_cursors, self.isolated_keystrokes, self.recorded_fontsizes):
            merged = dict()
            merged.update(vars(cursor))
            merged.update(vars(keystroke))
            merged['est_fontsize'] = font_size
            keystroke_data.append(merged)
        return keystroke_data

    def store_keystroke_timing(self, frame_id, isolated_keystroke):
        if len(self.keystroke_points) > 0:
            last_kpoint = self.keystroke_points[-1]
            last_xmin, _, _, _ = last_kpoint.last_detection_coordinates
            if abs(isolated_keystroke.kisolation_xmin - last_xmin) <= constants.DETECTION_SENSITIVITY:
                # print("Using Last KeystrokePoint with ID: ", last_kpoint.id)
                # print("Kunits: ", len(last_kpoint.kunits))
                last_kpoint.add_keystroke_unit(
                    frame_id, isolated_keystroke.get_isolation_coordinates(), isolated_keystroke)
                return last_kpoint

        kpoint = KeystrokePoint(
            frame_id, isolated_keystroke.get_isolation_coordinates())
        # print("Creating New KeystrokePoint with ID: ", kpoint.id)
        # print("Kunits: ", len(kpoint.kunits))
        kpoint.add_keystroke_unit(
            frame_id, isolated_keystroke.get_isolation_coordinates(), isolated_keystroke)
        self.keystroke_points.append(kpoint)
        return kpoint


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

    def __init__(self, detection_id, frame_id, score, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_w, bbox_h):
        self.detection_id = detection_id
        self.frame_id = frame_id
        self.score = score
        self.bbox_xmin = bbox_xmin
        self.bbox_xmax = bbox_xmax
        self.bbox_ymin = bbox_ymin
        self.bbox_ymax = bbox_ymax
        self.bbox_w = bbox_w
        self.bbox_h = bbox_h
        return


class IsolatedKeystroke(object):
    frame_id = -1
    kisolation_frame = None
    kisolation_w = 0
    kisolation_h = 0
    kisolation_xmin = 0
    kisolation_xmax = 0
    kisolation_ymin = 0
    kisolation_ymax = 0
    ocr_result = None

    def __init__(self, frame_id, kisolation_frame, kisolation_xmin, kisolation_xmax, kisolation_ymin, kisolation_ymax, kisolation_w, kisolation_h):
        self.frame_id = frame_id
        self.kisolation_frame = kisolation_frame
        self.kisolation_w = kisolation_w
        self.kisolation_h = kisolation_h
        self.kisolation_xmin = kisolation_xmin
        self.kisolation_xmax = kisolation_xmax
        self.kisolation_ymin = kisolation_ymin
        self.kisolation_ymax = kisolation_ymax
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

# KeystrokePoint class contains one or more KUnit (isolated_keystroke), and is used to train HMM for search-space reduction


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

    def add_keystroke_unit(self, frame_id, last_detection_coordinates, isolated_keystroke):
        self.k_vanish = frame_id
        self.last_detection_coordinates = last_detection_coordinates
        self.kunits.append(isolated_keystroke)

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


#initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


def save_keystroke_data(output_path, keystrokes):
    with open(output_path, mode='w') as csv_file:
        fieldnames = ['detection_id', 'score', 'frame_id', 'est_fontsize', 'bbox_xmin',
                      'bbox_xmax', 'bbox_ymin', 'bbox_ymax', 'bbox_w', 'bbox_h', 'kisolation_w', 'kisolation_h']
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(keystrokes)


def normalize_bbox_size(xmin, ymin, xmax, ymax):
    norm = 0.2
    return (xmin + (xmin * norm), ymin, xmax - (xmax * norm), ymax)


def pt_to_px(pt):
    return pt * constants.PT2PX_SIZE_FACTOR


def px_to_inch(px, PPI):
    return px / PPI


def get_cursor_height(cursor_ymax, cursor_ymin):
    return cursor_ymax - cursor_ymin

# automatically detect the font size of the letter based on cursor size


def calc_fontsize(cursor_ymax, cursor_ymin, PPI):
    cursor_height = get_cursor_height(cursor_ymax, cursor_ymin)
    font_size_inch = px_to_inch(cursor_height, PPI)
    font_size_pt = font_size_inch * constants.PT2INCH_SIZE_FACTOR
    return int(font_size_pt)


def calc_ppi(screen_w, screen_h, screen_size_inch):
    # from https://superuser.com/questions/1085734/how-do-i-know-the-dpi-of-my-laptop-screen
    # PPI = √(13662 + 7682) / 15.6 = 100.45
    PPI = sqrt(pow(screen_w, 2) + pow(screen_h, 2)) / screen_size_inch
    return PPI


def calc_average(arr):
    return sum(arr) / len(arr)


def get_video_size(video_path):
    vid = cv2.VideoCapture(video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    return (width, height)


def isolate_keystroke(frame, font_size, cursor_xmin, cursor_ymin, cursor_xmax, cursor_ymax, crop=False):
    image = Image.fromarray(frame)
    font_span = pt_to_px(font_size)

    # normalize the isolation box coordinate near the center of cursor bounding box
    cursor_x_center = (cursor_xmax - cursor_xmin) / 2

    xmin = (cursor_xmin - font_span) + cursor_x_center
    ymin = cursor_ymin
    xmax = cursor_xmin + (cursor_x_center)
    ymax = cursor_ymax

    crop_range = (xmin, ymin, xmax, ymax)

    isolation_width = xmax - xmin
    isolation_height = ymax - ymin

    if crop:
        # the bounding box will be cropped and returned as a frame
        frame = np.asarray(image.crop(crop_range))
    else:
        # the bounding box will be drawn inside the frame istead of cropped, returned   as a frame
        frame = draw_bbox(
            frame, crop_range[0], crop_range[1], crop_range[2], crop_range[3])

    return frame, crop_range, isolation_width, isolation_height


def do_OCR(keystroke, enhance=True, pad=True):
    im = keystroke.kisolation_frame
    # enhance the image before performing OCR
    # https://stackoverflow.com/questions/42566319/tesseract-ocr-reading-a-low-resolution-pixelated-font-esp-digits
    # https://stackoverflow.com/questions/9480013/image-processing-to-improve-tesseract-ocr-accuracy
    if enhance:
        # resize image
        RESIZE_FACTOR = 1.5
        im = cv2.resize(im, None, fx=RESIZE_FACTOR,
                        fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)

        # convert image to grayscale
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # applying dilation and erosion
        kernel = np.ones((1, 1), np.uint8)
        im = cv2.dilate(im, kernel, iterations=1)
        im = cv2.erode(im, kernel, iterations=1)

        # applying adaptive blur
        # im = cv2.adaptiveThreshold(cv2.bilateralFilter(im, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        # im = cv2.adaptiveThreshold(cv2.medianBlur(im, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        # applying normal blur
        im = cv2.threshold(cv2.medianBlur(im, 3), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # im = cv2.threshold(cv2.bilateralFilter(im, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # im = cv2.threshold(cv2.GaussianBlur(im, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
    return im, pytesseract.image_to_data(im, output_type=Output.DICT, config='--psm 10 --oem 3')
    


def draw_bbox(frame, xmin, ymin, xmax, ymax):
    color = colors[random.randint(0, len(colors) - 1)]
    color = [i * 255 for i in color]
    cv2.rectangle(frame, (int(xmin), int(ymin)),
                  (int(xmax), int(ymax)), color, 2)
    return frame


def frame_to_video(frames, output_path, w, h):
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()


def pad_image(image, target_size=100):
    padded_image = ImageOps.expand(image, target_size, 'black')
    return padded_image

# iterate through detected cursor while constantly estimating font size
# def extract_keystrokes_tracker(video_path):
#     camstroke = Camstroke()

#     vwidth, vheight = get_video_size(video_path)
#     PPI = calc_ppi(vwidth, vheight, screen_size_inch=13.3)

#     for i, tracked in enumerate(cursor_tracker.track_cursor(video_path, constants.WEIGHT_PATH, draw_bbox=True)):
#         frame, frame_num, xmin, ymin, xmax, ymax = tracked
#         im = Image.fromarray(frame)
#         font_size = calc_fontsize(ymax, ymin, PPI)

#         camstroke.last_cursor_position = (xmin, ymin, xmax, ymax)
#         camstroke.recorded_fontsizes.append(font_size)

#         keystroke_image = isolate_keystroke(frame, camstroke.get_avg_fontsize(), xmin, ymin, xmax, ymax, crop=True)
    # keystroke_image.show()
    # ocr = do_OCR(keystroke_image)
    # keystroke_image.save(fp="results/{}.png".format(frame_num))
    # print("Detected: ", ocr)


def extract_keystrokes_detector(video_path):
    camstroke = Camstroke()
    consecutive_streak = 0

    vwidth, vheight = get_video_size(video_path)
    PPI = calc_ppi(vwidth, vheight, screen_size_inch=13.3)

    for i, detected in enumerate(cursor_detector.detect_cursor(video_path, constants.WEIGHT_PATH, score_threshold=0.20)):
        frame, frame_id, pred_result = detected
        image_h, image_w, _ = frame.shape

        # for fast-testing
        # if frame_id >= 100:
        #     break

        boxes, scores, classes, valid_detections = pred_result
        if valid_detections[0] > 0:
            consecutive_streak += 1
            # print("Cons. Streak: ", consecutive_streak)
            for i in range(valid_detections[0]):
                if frame_id % 10 == 0:
                    print("Detection no. %d on Frame %d" % (i, frame_id))
                    # print("Score:", scores[0][i])

                coor = boxes[0][i]
                # print("Coor: ", coor)

                ymin = int(coor[0] * image_h)
                ymax = int(coor[2] * image_h)
                xmin = int(coor[1] * image_w)
                xmax = int(coor[3] * image_w)

                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                font_size = calc_fontsize(ymax, ymin, PPI)
                camstroke.last_cursor_position = (xmin, ymin, xmax, ymax)
                camstroke.recorded_fontsizes.append(font_size)

                detected_cursor = DetectedCursor(
                    i, frame_id, scores[0][i], xmin, ymin, xmax, ymax, bbox_w, bbox_h)

                isolated_frame, isolation_coordinate, isolated_width, isolated_height = isolate_keystroke(frame, camstroke.get_avg_fontsize(
                ), xmin, ymin, xmax, ymax, crop=True)  # change crop to False to draw isolation box instead of cropping it

                isolated_xmin, isolated_ymin, isolated_xmax, isolated_ymax = isolation_coordinate

                keystroke = IsolatedKeystroke(frame_id, isolated_frame, isolated_xmin,
                                              isolated_ymin, isolated_xmax, isolated_ymax, isolated_width, isolated_height)

                # save both detection and isolation data
                camstroke.detected_cursors.append(detected_cursor)
                camstroke.isolated_keystrokes.append(keystroke)

                keystroke_image, ocr_result = do_OCR(
                    keystroke, enhance=True, pad=False)
                # print("OCR Result: ", ocr_result)

                keystroke.ocr_result = ocr_result
                conf, keytext = keystroke.get_character()

                if keytext != None:
                    # print("Detected: ", temp)
                    # print("Isolation Coordinate (x, y): ", floor(keystroke.kisolation_xmin), floor(keystroke.kisolation_ymin))
                    keypoint = camstroke.store_keystroke_timing(
                        frame_id, keystroke)
                    timing_data = keypoint.get_timing_data()
                    # print(timing_data)

                # keystroke_image = keystroke.to_image()
                # keystroke_image.show()
                # keystroke_image.save(fp="results/{}_{}.png".format(frame_id, ocr_result))
        else:
            consecutive_streak = 0

    # save detection and isolation data to a file
    # save_keystroke_data('keystrokes.csv', camstroke.get_camstroke_detection_data())

    # save isolation bounding boxes to video format
    # frames = [f.kisolation_frame for f in camstroke.isolated_keystrokes]
    # frame_to_video(frames, 'output.avi', image_w, image_h)

    # store camstroke data for further processing
    utils.save_camstroke(camstroke, "results/camstroke.pkl")

    # pass data to hmm learning
    # keystroke_points = camstroke.keystroke_points
    # print(keystroke_points)
    # hmm.train(keystroke_points)


def loop_dataset():
    sizes = [14, 16, 18, 20, 22]
    for s in sizes:
        video_path = "../Recordings/vscode_font{}.mp4".format(s)
        print("Extracting from {}".format(video_path))
        extract_keystrokes_detector(video_path)


def detect_extract():
    video_path = "../Datasets/vscode_cut.mp4"
    print("Extracting from {}".format(video_path))
    extract_keystrokes_detector(video_path)


def learn_keystrokes():
    camstroke = utils.load_camstroke("results/camstroke.pkl")
    hmm.train(camstroke.keystroke_points)


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("mode", nargs='?', help="Run Camstroke in 'extract' or 'train' mode")

    # Read arguments from command line
    args = parser.parse_args()

    if args.mode == "extract":
        # loop_dataset()
        detect_extract()
    elif args.mode == "train":
        import hmm.hidden_markov as hmm
        learn_keystrokes()
