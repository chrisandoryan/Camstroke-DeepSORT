from PIL import Image, ImageOps
import cv2
from math import sqrt, floor
import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import itertools
import argparse

from text_processing import tesseract_ocr as OCR
from text_processing import cca

from helpers import constants
import helpers.utils as utils
from helpers.font import calc_font_width, calc_fontsize, get_cursor_height
from helpers.screen import px_to_inch, calc_ppi
from helpers.video import get_video_size

from dataclass.camstroke_data import Camstroke
from dataclass.keystroke import KUnit, KeystrokePoint
from dataclass.detected_cursor import DetectedCursor
from dataclass.isolated import IsolationWindow

from yolo_deepsort import cursor_tracker, cursor_detector
import hmm.viterbi_algorithm as viterbi

SCREEN_SIZE = 13.3 # in inch
PROPORTIONAL_FONT = "PROPORTIONAL"
FIXEDWIDTH_FONT = "FIXED-WIDTH"

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

def crop_isolation_window(frame, font_size, detection_coordinates, font_type, crop=False):
    cursor_xmin, cursor_ymin, cursor_xmax, cursor_ymax = detection_coordinates
    image = Image.fromarray(frame)
    # font_height = calc_font_height(font_size)
    font_width = calc_font_width(font_size)

    # normalize the isolation box coordinate near the center of cursor bounding box
    # based on font type (fixed-width/proportional)
    if font_type == PROPORTIONAL_FONT:
        cursor_x_center = (cursor_xmax - cursor_xmin) / 2

        xmin = (cursor_xmin - (2 * font_width)) # + (cursor_x_center * 0.5)
        ymin = cursor_ymin
        xmax = cursor_xmax # cursor_xmin + (cursor_x_center)
        ymax = cursor_ymax
    elif font_type == FIXEDWIDTH_FONT:
        cursor_x_center = (cursor_xmax - cursor_xmin) / 2

        xmin = (cursor_xmin - font_width) + cursor_x_center
        ymin = cursor_ymin
        xmax = cursor_xmin + (cursor_x_center * 0.8)
        ymax = cursor_ymax

    crop_range = (xmin, ymin, xmax, ymax)
    isolation_width = xmax - xmin
    isolation_height = ymax - ymin

    isolation_shape = (isolation_width, isolation_height)

    if crop:
        # the bounding box will be cropped and returned as a frame
        frame = np.asarray(image.crop(crop_range))
    else:
        # the bounding box will be drawn inside the frame istead of cropped, returned   as a frame
        frame = draw_bbox(frame, crop_range[0], crop_range[1], crop_range[2], crop_range[3])

    return frame, crop_range, isolation_shape


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

def run_with_yolo(video_path, font_type=FIXEDWIDTH_FONT):
    camstroke = Camstroke()
    consecutive_streak = 0

    vwidth, vheight = get_video_size(video_path)
    PPI = calc_ppi(vwidth, vheight, screen_size_inch=SCREEN_SIZE)

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
                # if frame_id % 10 == 0:
                #     print("Detection no. %d on Frame %d" % (i, frame_id))
                #     print("Score:", scores[0][i])

                coor = boxes[0][i]
                # print("Coor: ", coor)

                ymin = int(coor[0] * image_h)
                ymax = int(coor[2] * image_h)
                xmin = int(coor[1] * image_w)
                xmax = int(coor[3] * image_w)

                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                detection_coordinates = (xmin, ymin, xmax, ymax)
                detection_shape = (bbox_w, bbox_h)

                font_size = calc_fontsize(ymax, ymin, PPI)
                
                camstroke.last_cursor_position = detection_coordinates
                camstroke.recorded_fontsizes.append(font_size)

                detected_cursor = DetectedCursor(i, frame_id, scores[0][i], detection_coordinates, detection_shape)

                isolated_frame, isolation_coordinate, isolation_shape = crop_isolation_window(frame, camstroke.get_avg_fontsize(), detection_coordinates, font_type, crop=True)  # change crop to False to draw isolation box instead of cropping it
                isolation_window = IsolationWindow(frame_id, isolated_frame, isolation_coordinate, isolation_shape)

                # save both detection and isolation data
                camstroke.detected_cursors.append(detected_cursor)
                camstroke.isolation_windows.append(isolation_window)

                print("Frame ID: %d" % frame_id)
                print("Isolation Coordinate (x, y): ", floor(isolation_window.kisolation_xmin), floor(isolation_window.kisolation_ymin))
                print("Font Size (%s): %d" % (font_type, font_size))

                if font_type == PROPORTIONAL_FONT:
                    # perform connected component labelling (CCA)
                    keystroke_candidates, noises = cca.run_with_stats(isolation_window, font_size)
                    # print("Candidates for coordinate (x,y): ", xmin, ymin)
                    for c in keystroke_candidates:
                        kunit_bbox_x = c['coord']['x']
                        kunit_bbox_y = c['coord']['y']
                        kunit_bbox_w = c['shape']['w']
                        kunit_bbox_h = c['shape']['h']

                        kunit_coordinates = (kunit_bbox_x, kunit_bbox_y, kunit_bbox_x + kunit_bbox_w, kunit_bbox_y + kunit_bbox_h)
                        kunit_shape = (kunit_bbox_w, kunit_bbox_h)

                        kunit = KUnit(frame_id, c['mask'], kunit_coordinates, kunit_shape)
                        _, ocr_result = OCR.run_vanilla(c['mask'])

                        kunit.set_ocr_result(ocr_result)
                        print(kunit.get_character())

                        camstroke.store_kunit(kunit)
                        
                # elif font_type == FIXEDWIDTH_FONT:
                #     keystroke_image, ocr_result = OCR.run_advanced(isolation_window, enhance=True, pad=False)
                #     # print("OCR Result: ", ocr_result)

                #     isolation_window.ocr_result = ocr_result
                #     # conf, keytext = isolation_window.get_character()

                #     if keytext != None:
                #         print("Detected: ", keytext)
                #         # print("Isolation Coordinate (x, y): ", floor(isolation_window.kisolation_xmin), floor(isolation_window.kisolation_ymin))
                #         keypoint = camstroke.merge_keystroke_to_keystroke_points(frame_id, isolation_window)
                #         timing_data = keypoint.get_timing_data()
                #         # print(timing_data)

                # original/unprocessed image
                # keystroke_image = isolation_window.to_image()

                # keystroke_image.show()
                # keystroke_image.save(fp="results/{}_{}.png".format(frame_id, keytext))
        else:
            consecutive_streak = 0

    # save detection and isolation data to a file
    # save_keystroke_data('keystrokes.csv', camstroke.get_all_data())

    # save isolation bounding boxes to video format
    # frames = [f.kisolation_frame for f in camstroke.isolation_windows]
    # frame_to_video(frames, 'output.avi', image_w, image_h)

    # store camstroke data for further processing
    # utils.save_camstroke(camstroke, "results/camstroke.pkl")

    # pass data to hmm learning
    # keystroke_points = camstroke.keystroke_points
    # print(keystroke_points)
    # hmm.train(keystroke_points)


def loop_dataset():
    sizes = [14, 16, 18, 20, 22]
    for s in sizes:
        video_path = "../Recordings/vscode_font{}.mp4".format(s)
        print("Extracting from {}".format(video_path))
        run_with_yolo(video_path)


def detect_and_extract(font_type):
    video_path = "../Datasets/vscode_gfont2.mp4" # proportional font
    # video_path = "../Datasets/vscode_cut.mp4" # fixed-width font
    # print("Extracting from {}".format(video_path))
    run_with_yolo(video_path, font_type)


def train_and_predict():
    camstroke = utils.load_camstroke("results/camstroke.pkl")
    hmm_model, test_data = hmm.train(camstroke.keystroke_points)

    prediction = viterbi.predict(hmm_model, test_data)


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("mode", nargs='?',
                        help="Run Camstroke in 'extract' or 'train' mode")

    # Read arguments from command line
    args = parser.parse_args()

    if args.mode == "extract":
        # loop_dataset()
        font_type = PROPORTIONAL_FONT # run when the font width is proportional (e.g i has smaller width than z)
        # font_type = FIXEDWIDTH_FONT  # analyze when the font width is fixed (e.g i and z have same width)
        detect_and_extract(font_type)
    elif args.mode == "train":
        import hmm.hidden_markov as hmm
        train_and_predict()
