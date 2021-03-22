from helpers.utils import print_full
from helpers import constants
from helpers.hmm import preprocess
from camstroke import run_with_yolo

if __name__ == "__main__":
    screen_size = constants.SCREEN_SIZE
    font_type = constants.PROPORTIONAL_FONT 
    video_path = "../Datasets/keystroke_dynamic_forgery_1.mp4"

    camstroke = run_with_yolo(video_path, font_type, screen_size)
    dataset = preprocess(camstroke.keystroke_points)
    print_full(dataset)
