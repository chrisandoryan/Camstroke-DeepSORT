import cursor_tracker
from PIL import Image
import cv2
import pytesseract
from math import sqrt

# 1pt is 1/72 * inch
PT_SIZE = 72 

# how many cursor detections required to determine the font size
FONT_SIZE_CONSENSUS = 15

def px_to_inch(px, PPI):
    return px / PPI

def get_cursor_height(cursor_ymax, cursor_ymin):
    return cursor_ymax - cursor_ymin

# automatically detect the font size of the letter based on cursor size
def detect_fontsize(cursor_ymax, cursor_ymin, PPI):
    cursor_height = get_cursor_height(cursor_ymax, cursor_ymin)
    font_size_inch = px_to_inch(cursor_height, PPI)
    font_size_pt = font_size_inch * PT_SIZE
    return font_size_pt

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

def extract_keystroke(video_path):
    weight_path = "checkpoints/camstroke-yolov4-416"

    vwidth, vheight = get_video_size(video_path)
    PPI = calc_ppi(vwidth, vheight, screen_size_inch=13.3)

    font_sizes = []
    for i, tracked in enumerate(cursor_tracker.track_cursor(video_path, weight_path, draw_bbox=True)):
        if i == FONT_SIZE_CONSENSUS:
            break
        frame, xmin, ymin, xmax, ymax = tracked
        image = Image.fromarray(frame)

        font_size = detect_fontsize(ymax, ymin, PPI)
        print("Cursor Height #%s: %s" % (i, get_cursor_height(ymax, ymin)))
        print("Font Size #%s: %s" % (i, font_size))
        font_sizes.append(font_size)

    print("Video Path: ", video_path)
    print("Screen PPI: ", PPI)
    print("Average Font Size: ", calc_average(font_sizes))
    

def loop_dataset():
    sizes = [14, 16, 18, 20, 22]
    for s in sizes:
        video_path = "../Recordings/vscode_font{}.mp4".format(s)
        print("Extracting from {}".format(video_path))
        extract_keystroke(video_path)

loop_dataset()