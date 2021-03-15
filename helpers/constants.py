# 1pt is 1/72 * inch
# https://www.quora.com/How-is-font-size-measured
PT2INCH_SIZE_FACTOR = 72

# 1 pt is 1.328147px
# https://everythingfonts.com/font/tools/units/pt-to-px
PT2PX_SIZE_FACTOR = 1.328147 # 1

# how many cursor detections required to determine the average font size
FONT_SIZE_CONSENSUS = 100

# minimum confidence value for OCR detection
OCR_CONF_THRESHOLD = 10

# list of insignificant/invalid keystrokes that needs to be ignored
INVALID_KEYSTROKE = ["|", "="]

# path to weight for cursor tracker
WEIGHT_PATH = "./yolo_deepsort/checkpoints/camstroke-yolov4-416"

# if the next detected keystroke is +- around the last detected keystroke's x coordinate, the detection result is considered to be for the same detection attempt as the previous
DETECTION_SENSITIVITY = 5  # in pixels, alternatively we can use the font size