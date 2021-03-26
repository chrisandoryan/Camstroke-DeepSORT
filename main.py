import argparse
from helpers.utils import print_full, load_camstroke, save_camstroke, make_dirs, print_info
from helpers import constants
from helpers.data import preprocess
from helpers.keylog import plotDDT, plotDUT
from helpers.video import get_fps
from camstroke import run_with_yolo
from keylogger import keylogger

screen_size = constants.DEFAULT_SCREEN_SIZE
font_type = constants.PROPORTIONAL_FONT 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs='?',
                        help="Run Camstroke in 'extract' or 'train' mode")

    TEST_NUMBER = 5
    EXPERIMENT_DIR = "./results/experiments/test_%d" % TEST_NUMBER

    args = parser.parse_args()
    if args.mode == "keylogger":
        print_info("Preparing experiment directory at %s" % EXPERIMENT_DIR)
        make_dirs(EXPERIMENT_DIR)

        print_info("Firing up Keylogger engine")
        keylogger.run(save_path="%s/keylog_result.csv" % EXPERIMENT_DIR)

    elif args.mode == "camstroke":
        video_path = "../Datasets/keystroke_dynamic_forgery_4.mp4"
        camstroke = run_with_yolo(video_path, font_type, screen_size)
        save_camstroke(camstroke, save_path="%s/camstroke.pkl")
        
        keypoints = camstroke.get_kpoints()
        dataset = preprocess(keypoints)
        dataset.to_csv("%s/camstroke_kpoints.csv")

    elif args.mode == "analyze":
        camstroke = load_camstroke(save_path="%s/camstroke.pkl")
        keypoints = camstroke.get_kpoints()

        plotDDT(keypoints)
        plotDUT(keypoints)

        