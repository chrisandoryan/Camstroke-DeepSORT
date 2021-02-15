import cursor_tracker
import cursor_detector
from PIL import Image, ImageOps
import cv2
import pytesseract
from math import sqrt
import matplotlib.pyplot as plt
import random
import numpy as np
import csv

class Camstroke(object):
    last_pos = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax
    recorded_fontsizes = []
    detected_cursors = []
    isolated_keystrokes = []

    def get_avg_fontsize(self):
        return calc_average(self.recorded_fontsizes)

    def get_keystroke_data(self):
        keystroke_data = []
        for cursor, keystroke, font_size in zip(self.detected_cursors, self.isolated_keystrokes, self.recorded_fontsizes):
            merged = dict()
            merged.update(vars(cursor))
            merged.update(vars(keystroke))
            merged['est_fontsize'] = font_size
            keystroke_data.append(merged)
        return keystroke_data

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
    kisolation_frame = None
    kisolation_w = 0
    kisolation_h = 0
    def __init__(self, kisolation_frame, kisolation_w, kisolation_h):
        self.kisolation_frame = kisolation_frame
        self.kisolation_w = kisolation_w
        self.kisolation_h = kisolation_h
        return
    def to_image(self):
        return Image.fromarray(self.kisolation_frame)


#initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# 1pt is 1/72 * inch
# https://www.quora.com/How-is-font-size-measured
PT2INCH_SIZE_FACTOR = 72

# 1 pt is 1.328147px
# https://everythingfonts.com/font/tools/units/pt-to-px
PT2PX_SIZE_FACTOR = 1  # 1.328147

# how many cursor detections required to determine the average font size
FONT_SIZE_CONSENSUS = 100

# path to weight for cursor tracker
WEIGHT_PATH = "checkpoints/camstroke-yolov4-416"

def save_keystroke_data(output_path, keystrokes):
    with open(output_path, mode='w') as csv_file:
        fieldnames = ['detection_id', 'score', 'frame_id', 'est_fontsize', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax', 'bbox_w', 'bbox_h', 'kisolation_w', 'kisolation_h']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(keystrokes)
        
def normalize_bbox_size(xmin, ymin, xmax, ymax):
    norm = 0.2
    return (xmin + (xmin * norm), ymin, xmax - (xmax * norm), ymax)

def pt_to_px(pt):
    return pt * PT2PX_SIZE_FACTOR

def px_to_inch(px, PPI):
    return px / PPI

def get_cursor_height(cursor_ymax, cursor_ymin):
    return cursor_ymax - cursor_ymin

# automatically detect the font size of the letter based on cursor size
def calc_fontsize(cursor_ymax, cursor_ymin, PPI):
    cursor_height = get_cursor_height(cursor_ymax, cursor_ymin)
    font_size_inch = px_to_inch(cursor_height, PPI)
    font_size_pt = font_size_inch * PT2INCH_SIZE_FACTOR
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
        frame =  draw_bbox(frame, crop_range[0], crop_range[1], crop_range[2], crop_range[3])

    return frame, isolation_width, isolation_height

def do_OCR(keystroke, enhance=True, pad=True):
    im = keystroke.kisolation_frame
    # enhance the image before performing OCR
    # https://stackoverflow.com/questions/42566319/tesseract-ocr-reading-a-low-resolution-pixelated-font-esp-digits
    # https://stackoverflow.com/questions/9480013/image-processing-to-improve-tesseract-ocr-accuracy
    if enhance:
        # resize image
        RESIZE_FACTOR = 2.5
        im = cv2.resize(im, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)

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
        im = cv2.threshold(cv2.medianBlur(im, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        im = cv2.threshold(cv2.bilateralFilter(im, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    im = Image.fromarray(im)

    # perform image pad and resize for higher resolution
    if pad:
        im = pad_image(im, target_size=50)

    return im, pytesseract.image_to_string(im, config='--psm 10 --oem 1').strip()
    # return im, pytesseract.image_to_data(im, config='--psm 10 --oem 3')

def draw_bbox(frame, xmin, ymin, xmax, ymax):
    color = colors[random.randint(0, len(colors) - 1)]
    color = [i * 255 for i in color]
    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    return frame

def frame_to_video(frames, output_path, w, h):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))
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

#     for i, tracked in enumerate(cursor_tracker.track_cursor(video_path, WEIGHT_PATH, draw_bbox=True)):
#         frame, frame_num, xmin, ymin, xmax, ymax = tracked
#         im = Image.fromarray(frame)
#         font_size = calc_fontsize(ymax, ymin, PPI)

#         camstroke.last_pos = (xmin, ymin, xmax, ymax)
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

    for i, detected in enumerate(cursor_detector.detect_cursor(video_path, WEIGHT_PATH, score_threshold=0.20)):
        frame, frame_id, pred_result = detected
        image_h, image_w, _ = frame.shape
        
        # for fast-testing
        # if frame_id >= 500:
        #     break

        boxes, scores, classes, valid_detections = pred_result
        if valid_detections[0] > 0:
            consecutive_streak += 1
            print("Cons. Streak: ", consecutive_streak)
            for i in range(valid_detections[0]):
                print("Detection no. %d on Frame %d" % (i, frame_id))
                print("Score:", scores[0][i])

                coor = boxes[0][i]
                print("Coor: ", coor)

                ymin = int(coor[0] * image_h)
                ymax = int(coor[2] * image_h)
                xmin = int(coor[1] * image_w)
                xmax = int(coor[3] * image_w)

                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                font_size = calc_fontsize(ymax, ymin, PPI)
                camstroke.last_pos = (xmin, ymin, xmax, ymax)
                camstroke.recorded_fontsizes.append(font_size)

                detected_cursor = DetectedCursor(i, frame_id, scores[0][i], xmin, ymin, xmax, ymax, bbox_w, bbox_h)

                isolated_frame, isolated_width, isolated_height = isolate_keystroke(frame, camstroke.get_avg_fontsize(), xmin, ymin, xmax, ymax, crop=True) # change crop to False to draw isolation box instead of cropping it

                keystroke = IsolatedKeystroke(isolated_frame, isolated_width, isolated_height)

                # save both detection and isolation data
                camstroke.detected_cursors.append(detected_cursor)
                camstroke.isolated_keystrokes.append(keystroke)

                keystroke_image, ocr_result = do_OCR(keystroke, enhance=True, pad=False)
                print(ocr_result)

                # keystroke_image = keystroke.to_image()
                # keystroke_image.show()
                keystroke_image.save(fp="results/{}_{}.png".format(frame_id, ocr_result))
        else:
            consecutive_streak = 0
    
    # save_keystroke_data('keystrokes.csv', camstroke.get_keystroke_data())
    frames = [f.kisolation_frame for f in camstroke.isolated_keystrokes]
    # frame_to_video(frames, 'output.avi', image_w, image_h)

def loop_dataset():
    sizes = [14, 16, 18, 20, 22]
    for s in sizes:
        video_path = "../Recordings/vscode_font{}.mp4".format(s)
        print("Extracting from {}".format(video_path))
        extract_keystrokes_detector(video_path)

def main():
    video_path = "../Datasets/vscode_cut.mp4"
    print("Extracting from {}".format(video_path))
    extract_keystrokes_detector(video_path)

if __name__ == '__main__':
    # loop_dataset()
    main()