from helpers.utils import print_full, load_camstroke
from helpers import constants
from helpers.hmm import preprocess
from helpers.keylog import plotDDT, plotDUT
from helpers.video import get_fps
from camstroke import run_with_yolo

if __name__ == "__main__":
    screen_size = constants.SCREEN_SIZE
    font_type = constants.PROPORTIONAL_FONT 
    video_path = "../Datasets/keystroke_dynamic_forgery_3.mp4"
    fps = get_fps(video_path)

    camstroke = run_with_yolo(video_path, font_type, screen_size)
    camstroke = load_camstroke("results/experiments/test_3/camstroke_forgery_attack.pkl")

    plotDDT([x.get_timing_data(fps) for x in camstroke.keystroke_points])
    plotDUT([x.get_timing_data(fps) for x in camstroke.keystroke_points])
    dataset = preprocess(fps, camstroke.keystroke_points)
    print_full(dataset)
