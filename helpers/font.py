from helpers import constants
from helpers.screen import px_to_inch

# https://medium.com/@zkareemz/golden-ratio-62b3b6d4282a
def calc_font_height(pt):
    return pt * constants.PT2PX_SIZE_FACTOR


def calc_font_width(pt):
    return pt / constants.PT2PX_SIZE_FACTOR


def get_cursor_height(cursor_ymax, cursor_ymin):
    return cursor_ymax - cursor_ymin

# automatically detect the font size of the letter based on cursor size
def calc_fontsize(cursor_ymax, cursor_ymin, PPI):
    cursor_height = get_cursor_height(cursor_ymax, cursor_ymin)
    font_size_inch = px_to_inch(cursor_height, PPI)
    font_size_pt = font_size_inch * constants.PT2INCH_SIZE_FACTOR
    return int(font_size_pt)
