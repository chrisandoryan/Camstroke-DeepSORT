import cv2
from PIL import Image, ImageOps
import numpy as np
from helpers.font import calc_font_height, calc_font_width
import time

"""
Performs connected component labelling (CCA/CCL) for isolating characters from image.
Algorithm v1.0:
1. perform CCA and isolate character regions
2. if there is more than 2 regions detected (indicates that multiple regions other than background is detected), then:
    - eliminate regions that are cut off (the bbox's x or y is intersected with the border of the image)
    - 
3. if there is two object with exact same X (indicates both aligned vertically), combine both (for i and j case)
4. the typed character is the rightmost isolated region in the frame (karena karakter terakhir pasti berada tepat sebelum kursor, dimana kursor berada di kanan)
"""

CONNECTIVITY = 8


def the_algorithm(im, output):
    """
    Algorithm v2.0
    1. Skip the first region (since it is the background)
    2. If there are only one region left (after removing the background) skip it (it must be the cursor since the cursor must always appear on the frame by design)
    3. If there are more than 2 regions, differentiate each region into specific character components (assumptions):
    3.1 Tallest region (region with highest height) must be the cursor (CURSOR_REGION)
    3.2 Rightmost region except the cursor/CURSOR_REGION must be the last typed character
    3.3 The rest can be considered as noise
    """
    (numLabels, labels, stats, centroids) = output
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im_height, im_width, _ = im.shape

    print("Stats: ", stats)

    if numLabels <= 2:
        print("Skipping since only 2 regions detected")
    else:
        # get background region (assumption 1)
        background_index = 0
        background = stats[background_index]
        stats = np.delete(stats, background_index, axis=0)
        print("BG: ", background)
        print("Stats: ", stats)
        draw_bbox(im, background_index, background, centroids[background_index], labels)

        # get cursor region (assumption 3.1)
        cursor_region_index = np.where(
            stats[:, cv2.CC_STAT_HEIGHT] == np.amax(stats[:, cv2.CC_STAT_HEIGHT]))
        cursor_region = stats[cursor_region_index][0]
        stats = np.delete(stats, cursor_region_index, axis=0)

        print("CR: ", cursor_region)
        print("Stats: ", stats)
        draw_bbox(im, cursor_region_index, cursor_region, centroids[cursor_region_index + 1], labels)

        # get last typed character (assumption 3.2).
        # filter indices where height is below cursor_region's height
        candidates = stats[stats[:, cv2.CC_STAT_HEIGHT]
                           < cursor_region[cv2.CC_STAT_HEIGHT]]
        # then filter where value of x coordinate that is bigger than other regions (the rightmost)
        rightmost_index = np.where(
            candidates[:, cv2.CC_STAT_LEFT] == np.amax(stats[:, cv2.CC_STAT_LEFT]))
        keystroke_region = stats[rightmost_index][0]
        stats = np.delete(stats, rightmost_index, axis=0)
        print("KR: ", keystroke_region)
        draw_bbox(im, rightmost_index, keystroke_region, centroids[rightmost_index + 2], labels)

    return


def draw_bbox(im, i, stat, centroid, labels):
    bbox_x = stat[cv2.CC_STAT_LEFT]
    bbox_y = stat[cv2.CC_STAT_TOP]
    bbox_w = stat[cv2.CC_STAT_WIDTH]
    bbox_h = stat[cv2.CC_STAT_HEIGHT]
    area = stat[cv2.CC_STAT_AREA]
    (bbox_cX, bbox_cY) = centroid

    cloned_frame = im.copy()
    cv2.rectangle(cloned_frame, (bbox_x, bbox_y),
                  (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 3)
    cv2.circle(cloned_frame, (int(bbox_cX), int(bbox_cY)),
               4, (0, 0, 255), -1)

    # construct a mask for the current connected component by
    # finding a pixels in the labels array that have the current
    # connected component ID
    componentMask = (labels == i).astype("uint8") * 255

    # show our output image and connected component mask
    cv2.imshow("Output", cloned_frame)
    cv2.imshow("Connected Component", componentMask)
    cv2.waitKey(0)


# function to display isolated regions (for debugging purposes)
def display_regions(im, font_size, output):
    (numLabels, labels, stats, centroids) = output
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im_height, im_width, _ = im.shape

    font_width = calc_font_width(font_size)
    font_height = calc_font_height(font_size)

    print("Font Width: ", font_width)
    print("Font Height: ", font_height)

    print(stats)

    # loop over the number of unique connected component labels
    # skip the first result since it usually is the background being isolated
    # skip the second result since it usually is the cursor
    for i in range(2, numLabels):
        # extract the connected component statistics and centroid for
        # the current label
        bbox_x = stats[i, cv2.CC_STAT_LEFT]
        bbox_y = stats[i, cv2.CC_STAT_TOP]
        bbox_w = stats[i, cv2.CC_STAT_WIDTH]
        bbox_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (bbox_cX, bbox_cY) = centroids[i]

        print("W", bbox_w)
        print("H", bbox_h)
        print("Area", area)
        print("X", bbox_x)
        print("Y", bbox_y)

        # clone our original image (so we can draw on it) and then draw
        # a bounding box surrounding the connected component along with
        # a circle corresponding to the centroid
        cloned_frame = im.copy()
        cv2.rectangle(cloned_frame, (bbox_x, bbox_y),
                      (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 3)
        cv2.circle(cloned_frame, (int(bbox_cX), int(bbox_cY)),
                   4, (0, 0, 255), -1)

        # construct a mask for the current connected component by
        # finding a pixels in the labels array that have the current
        # connected component ID
        componentMask = (labels == i).astype("uint8") * 255

        # show our output image and connected component mask
        cv2.imshow("Output", cloned_frame)
        cv2.imshow("Connected Component", componentMask)
        cv2.waitKey(0)


def run_with_stats(im, font_size):
    im = np.array(im)

    # apply connected component analysis to the thresholded image
    cca_result = cv2.connectedComponentsWithStats(im, CONNECTIVITY, cv2.CV_32S)
    # display_regions(im, font_size, result)
    the_algorithm(im, cca_result)
