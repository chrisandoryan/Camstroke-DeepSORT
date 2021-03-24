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
from helpers.video import get_video_size, frame_to_video, get_fps
from helpers.image import perform_watershed, solve_overlapping

from dataclass.camstroke_data import Camstroke
from dataclass.keystroke import KUnit, KeystrokePoint
from dataclass.detected_cursor import DetectedCursor
from dataclass.isolation_window import IsolationWindow

from yolo_deepsort import cursor_tracker, cursor_detector
import keystroke_prediction.viterbi_algorithm as viterbi

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
    if font_type == constants.PROPORTIONAL_FONT:
        cursor_x_center = (cursor_xmax - cursor_xmin) / 2

        xmin = (cursor_xmin - (2 * font_width)) # + (cursor_x_center * 0.5)
        ymin = cursor_ymin
        xmax = cursor_xmax # cursor_xmin + (cursor_x_center)
        ymax = cursor_ymax
    elif font_type == constants.FIXEDWIDTH_FONT:
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

def to_absolute_coordinates(isolation_coordinates, kunit_coordinates):
    # print(isolation_coordinates)
    # print(kunit_coordinates)

    iso_xmin, iso_ymin, iso_xmax, iso_ymax = isolation_coordinates
    kun_xmin, kun_ymin, kun_xmax, kun_ymax = kunit_coordinates

    # descale the pixels (normalize)
    kun_xmin = kun_xmin / constants.RESIZE_FACTOR
    kun_ymin = kun_ymin / constants.RESIZE_FACTOR
    kun_xmax = kun_xmax / constants.RESIZE_FACTOR
    kun_ymax = kun_ymax / constants.RESIZE_FACTOR

    abs_xmin = iso_xmin + kun_xmin
    abs_ymin = iso_ymin + kun_ymin
    abs_xmax = iso_xmin + kun_xmax
    abs_ymax = iso_ymin + kun_ymax

    abs_coordinates = (abs_xmin, abs_ymin, abs_xmax, abs_ymax)
    return abs_coordinates

def check_cursor_backwards(coords, last_coords):
    xmin, ymin, xmax, ymax = coords
    last_xmin, last_ymin, last_xmax, last_ymax = last_coords
    return xmin - last_xmin < 0 and xmin - last_xmin <= -constants.DETECTION_SENSITIVITY

def map_cursor_movements(coords, last_coords, font_size):
    # TODO: map cursor movements 
    # backwards, new line, teleporting
    font_height = calc_font_height(font_size)
    font_width = calc_font_width(font_size)

    xmin, ymin, xmax, ymax = coords
    last_xmin, last_ymin, last_xmax, last_ymax = last_coords

    cursor_backward = xmin - last_xmin < 0 and xmin - last_xmin <= -constants.DETECTION_SENSITIVITY
    # cursor_teleport = xmin - last_xmin >= font_width or ymin 
    # cursor_newline

    return

def run_with_yolo(video_path, font_type=constants.FIXEDWIDTH_FONT, screen_size=constants.SCREEN_SIZE):
    camstroke = Camstroke()
    consecutive_streak = 0

    vwidth, vheight = get_video_size(video_path)
    PPI = calc_ppi(vwidth, vheight, screen_size_inch=screen_size)
    fps = get_fps(video_path)

    print("Video Size: %s x %s" % (vwidth, vheight))
    print("Video FPS: %s" % fps)
    print("PPI: %s " % PPI)

    for i, detected in enumerate(cursor_detector.detect_cursor(video_path, constants.WEIGHT_PATH, score_threshold=0.00)):
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
                    print("Score:", scores[0][i])

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

                if check_cursor_backwards(detection_coordinates, camstroke.last_cursor_position):
                    print("Delete Event detected")
                
                camstroke.last_cursor_position = detection_coordinates
                camstroke.recorded_fontsizes.append(font_size)

                detected_cursor = DetectedCursor(i, frame_id, scores[0][i], detection_coordinates, detection_shape)

                isolated_frame, isolation_coordinates, isolation_shape = crop_isolation_window(frame, camstroke.get_avg_fontsize(), detection_coordinates, font_type, crop=True)  # change crop to False to draw isolation box instead of cropping it
                isolation_window = IsolationWindow(frame_id, isolated_frame, isolation_coordinates, isolation_shape)

                # save both detection and isolation data
                camstroke.detected_cursors.append(detected_cursor)
                camstroke.isolation_windows.append(isolation_window)

                if font_type == constants.PROPORTIONAL_FONT:
                    # perform connected component labelling (CCA)
                    keystroke_candidates, noises = cca.run_with_stats(isolation_window, font_size)
                    # print("Candidates for coordinate (x,y): ", xmin, ymin)
                    for c in keystroke_candidates:
                        # perform intersection solving if c type is tallest region (to try to split character that interconnected with the cursor)
                        if c['type'] == constants.TALLEST_TYPE:
                            kunit_image = solve_overlapping(c['mask'])
                        else:
                            kunit_image = c['mask']

                        kunit_bbox_x = c['coord']['x']
                        kunit_bbox_y = c['coord']['y']
                        kunit_bbox_w = c['shape']['w']
                        kunit_bbox_h = c['shape']['h']

                        kunit_coordinates = (kunit_bbox_x, kunit_bbox_y, kunit_bbox_x + kunit_bbox_w, kunit_bbox_y + kunit_bbox_h)
                        kunit_shape = (kunit_bbox_w, kunit_bbox_h)

                        # convert kunit coordinate relative to the video frame, instead of relative to the isolation bbox frame
                        kunit_absolute_coordinates = to_absolute_coordinates(isolation_coordinates, kunit_coordinates)

                        kunit = KUnit(frame_id, kunit_image, c['type'], kunit_absolute_coordinates, kunit_shape)
                        _, ocr_result = OCR.run_vanilla(kunit_image)

                        kunit.set_ocr_result(ocr_result)

                        # print("Frame ID: %d" % frame_id)
                        # print("Isolation Coordinate: ", isolation_coordinates)
                        # print("Font Size (%s): %d" % (font_type, font_size))
                        # print("KUnit Absolute Coord: ", kunit_absolute_coordinates)
                        # print("OCR Result: ", kunit.get_character())
                        # print()

                        kpoint = camstroke.store_kunit(frame_id, kunit)
                        timing_data = kpoint.get_timing_data(fps)
                        print("Timing Data: ", timing_data)
                        # input()
                        
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
    utils.save_camstroke(camstroke, "results/experiments/test_3/camstroke_forgery_attack.pkl")

    # pass data to hmm learning
    # keystroke_points = camstroke.keystroke_points
    # print(keystroke_points)
    # hmm.train(keystroke_points)

    print("Done.")
    return camstroke


def loop_dataset():
    sizes = [14, 16, 18, 20, 22]
    for s in sizes:
        video_path = "../Recordings/vscode_font{}.mp4".format(s)
        print("Extracting from {}".format(video_path))
        run_with_yolo(video_path)


def _detect_and_extract(font_type, screen_size):
    video_path = "../Datasets/vscode_gfont2.mp4" # proportional font
    # video_path = "../Datasets/vscode_cut.mp4" # fixed-width font
    # print("Extracting from {}".format(video_path))
    run_with_yolo(video_path, font_type, screen_size)


def _train_and_predict():
    camstroke = utils.load_camstroke("results/pickles/camstroke_cca.pkl")
    # pohmm_model, test_data = pohmm.train(camstroke.keystroke_points)
    # hmm_model = hmm.train(camstroke.keystroke_points)

    # prediction = viterbi.predict(pohmm_model, test_data)


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
        # PROPORTIONAL: run when the font width is proportional (e.g i has smaller width than z)
        # FIXED-WIDTH: analyze when the font width is fixed (e.g i and z have same width)
        screen_size = float(input("Enter your Screen Size (inch): "))
        font_type = constants.PROPORTIONAL_FONT 
        _detect_and_extract(font_type, screen_size)
    elif args.mode == "train":
        import keystroke_prediction.pohmm as pohmm
        import keystroke_prediction.hmm as hmm
        _train_and_predict()
