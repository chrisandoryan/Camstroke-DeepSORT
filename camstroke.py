import cursor_tracker
import cursor_detector
from PIL import Image
import cv2
import pytesseract
from math import sqrt
import matplotlib.pyplot as plt
import random
import numpy as np

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
    crop_range = (cursor_xmin - font_span, cursor_ymin,
                  cursor_xmin, cursor_ymax)
    if crop:
        return image.crop(crop_range)
    else:
        return Image.fromarray(draw_bbox(frame, crop_range[0], crop_range[1], crop_range[2], crop_range[3]))


def do_ocr(im):
    return pytesseract.image_to_string(im)


def draw_bbox(frame, xmin, ymin, xmax, ymax):
    color = colors[random.randint(0, len(colors) - 1)]
    color = [i * 255 for i in color]
    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    return frame


class Camstroke:
    last_pos = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax
    recorded_fontsizes = []

    def get_fontsize(self):
        return calc_average(self.recorded_fontsizes)

# iterate through detected cursor while constantly estimating font size
def extract_keystrokes_tracker(video_path):
    camstroke = Camstroke()

    vwidth, vheight = get_video_size(video_path)
    PPI = calc_ppi(vwidth, vheight, screen_size_inch=13.3)

    for i, tracked in enumerate(cursor_tracker.track_cursor(video_path, WEIGHT_PATH, draw_bbox=True)):
        frame, frame_num, xmin, ymin, xmax, ymax = tracked
        im = Image.fromarray(frame)
        font_size = calc_fontsize(ymax, ymin, PPI)

        camstroke.last_pos = (xmin, ymin, xmax, ymax)
        camstroke.recorded_fontsizes.append(font_size)

        keystroke_image = isolate_keystroke(frame, camstroke.get_fontsize(), xmin, ymin, xmax, ymax, crop=False)
        # keystroke_image.show()
        # ocr = do_ocr(keystroke_image)
        keystroke_image.save(fp="results/{}.png".format(frame_num))
        # print("Detected: ", ocr)

def extract_keystrokes_detector(video_path):
    camstroke = Camstroke()
    consecutive_streak = 0

    vwidth, vheight = get_video_size(video_path)
    PPI = calc_ppi(vwidth, vheight, screen_size_inch=13.3)

    for i, detected in enumerate(cursor_detector.detect_cursor(video_path, WEIGHT_PATH, score_threshold=0.15)):
        frame, frame_id, pred_result = detected
        image_h, image_w, _ = frame.shape

        boxes, scores, classes, valid_detections = pred_result
        if valid_detections[0] > 0:
            consecutive_streak += 1
            print("Cons. Streak: ", consecutive_streak)
            for i in range(valid_detections[0]):
                print("Detection no. %d on Frame %d" % (i, frame_id))
                print("Scores:", scores[0][i])

                coor = boxes[0][i]
                print("Coor: ", coor)
                ymin = int(coor[0] * image_h)
                ymax = int(coor[2] * image_h)
                xmin = int(coor[1] * image_w)
                xmax = int(coor[3] * image_w)

                font_size = calc_fontsize(ymax, ymin, PPI)
                camstroke.last_pos = (xmin, ymin, xmax, ymax)
                camstroke.recorded_fontsizes.append(font_size)

                keystroke_image = isolate_keystroke(frame, camstroke.get_fontsize(), xmin, ymin, xmax, ymax, crop=False)
                keystroke_image.save(fp="results/{}.png".format(frame_id))
        else:
            consecutive_streak = 0

def loop_dataset():
    sizes = [14, 16, 18, 20, 22]
    for s in sizes:
        # video_path = "../Recordings/vscode_font{}.mp4".format(s)
        video_path = "../Datasets/vscode.mp4"
        print("Extracting from {}".format(video_path))
        extract_keystrokes_detector(video_path)

loop_dataset()
