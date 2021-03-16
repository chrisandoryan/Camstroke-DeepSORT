import cv2
from PIL import Image, ImageOps
import numpy as np
from helpers.font import calc_font_height, calc_font_width
from helpers.image import enhance_image
from helpers.utils import unique_array_dict
import time

STACKED_REGION_THRESHOLD = 2 # if two regions is 2 pixels adrift between each other, consider those regions belongs to the same chacter (e.g i and j)

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


def create_region_object(i, stats, centroids, empty=False):
    region_data = {
        "type": "",
        "index": -1,
        "shape": {
                "w": -1,
                "h": -1,
                "area": -1
        },
        "coord": {
            "x": -1,
            "y": -1
        },
        "centroid": {
            "cX": -1,
            "cY": -1
        }
    }
    if empty:
        return region_data
    else:
        bbox_x = stats[i, cv2.CC_STAT_LEFT]
        bbox_y = stats[i, cv2.CC_STAT_TOP]
        bbox_w = stats[i, cv2.CC_STAT_WIDTH]
        bbox_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (bbox_cX, bbox_cY) = centroids[i]

        region_data['index'] = i
        region_data['shape'] = {
            "w": bbox_w,
            "h": bbox_h,
            "area": area
        }
        region_data['coord'] = {
            "x": bbox_x,
            "y": bbox_y
        }
        region_data['centroid'] = {
            "cX": bbox_cX,
            "cY": bbox_cY
        }
        return region_data

def sort_by_x_position(candidates):
    return sorted(candidates, key=lambda k: k['coord']['x']) 

def detect_stacked_regions(candidates):
    prev = candidates[0]
    prev_xmin = -1000
    prev_xmax = -1000
    for c in candidates:
        xmin = c['coord']['x']
        xmax = c['coord']['x'] + c['shape']['w']
        if abs(prev_xmin - xmin) <= STACKED_REGION_THRESHOLD or abs(prev_xmax - xmax) < STACKED_REGION_THRESHOLD:
            yield prev['index'], c['index']
        prev = c
        prev_xmin = xmin
        prev_xmax = xmax

def the_algorithm(im, output):
    """
    Algorithm v2.0
    1. Skip the first region (since it is the background)
    2. If there are only one region left (after removing the background) skip it (it must be the cursor since the cursor must always appear on the frame by design)
    3. If there are more than 2 regions, differentiate each region into specific character components (assumptions):
    3.1 Tallest region (region with highest height) must be the cursor (CURSOR_REGION)
    3.2 Rightmost region except the cursor/CURSOR_REGION must be the last typed character
    3.2 (alt) Every region to the left of the CURSOR_REGION that is not intersect with the border
    3.3 The rest can be considered as noise
    """
    (numLabels, labels, stats, centroids) = output
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im_height, im_width, _ = im.shape

    # since both height and width starts in 1 instead of 0, we subtract 1 from them
    # im_height = im_height - 1
    # im_width = im_width - 1

    print("imw", im_width)
    print("imh", im_height)

    noises = []
    candidates = []

    if numLabels <= 2:
        print("Skipping since only 2 regions detected")
        pass
    else:
        # get background region (assumption 1)
        background_index = 0
        background = stats[background_index]
        # print("BG: ", background)

        tallest_region = create_region_object(1, stats, centroids, empty=True)
        rightmost_region = create_region_object(1, stats, centroids, empty=True)

        for i in range(1, numLabels):
            bbox_x = stats[i, cv2.CC_STAT_LEFT]
            bbox_w = stats[i, cv2.CC_STAT_WIDTH]
            bbox_h = stats[i, cv2.CC_STAT_HEIGHT]

            # get the tallest region / cursor region (assumption 3.1)
            if bbox_h > tallest_region['shape']['h']:
                tallest_region = create_region_object(i, stats, centroids, empty=False)
                tallest_region['type'] = "tallest"
                # print("Updating TR: ", tallest_region)
            
            # get the rightmost region (assumption 3.2)
            if bbox_x >= rightmost_region['coord']['x'] and bbox_h < tallest_region['shape']['h']:
                rightmost_region = create_region_object(i, stats, centroids, empty=False)
                rightmost_region['type'] = "rightmost"    
                # print("Updating RR: ", rightmost_region)

            # get all regions to the left of cursor_region, not intersecting with border (assumption 3.2 alt.)
            if bbox_h < tallest_region['shape']['h'] and (bbox_x + bbox_w < im_width and bbox_x > 0):
                candidate_region = create_region_object(i, stats, centroids, empty=False)
                candidate_region['type'] = "candidate"    
                candidates.append(candidate_region)
            else:
                noise_region = create_region_object(i, stats, centroids, empty=False)
                noise_region['type'] = "noise"
                noises.append(noise_region)
        
        candidates.append(rightmost_region)
        candidates = unique_array_dict(candidates, "index")

        # sort based on x coordinate
        candidates = sort_by_x_position(candidates)

        # merge vertically aligned regions (to resolve i and j detection limitation)
        # if found, we change labels of both regions into the same label (i)
        # and remove the data with old label from list of candidates
        for a_index, b_index in detect_stacked_regions(candidates):
            candidates = [c for c in candidates if c['index'] != a_index]
            labels[labels == a_index] = b_index

        # TODO: noises hasn't been processed with stacked regions detection
        noises.append(tallest_region)
        noises = unique_array_dict(noises, "index")
        
        for c in candidates:
            c['mask'] = (labels == c['index']).astype("uint8") * 255
            # draw_bbox(im, c['index'], stats[c['index']], centroids[c['index']], labels, "Candidate")
        for n in noises:
            n['mask'] = (labels == n['index']).astype("uint8") * 255

    return candidates, noises


def draw_bbox(im, i, stat, centroid, labels, frame_label="Output"):
    bbox_x = stat[cv2.CC_STAT_LEFT]
    bbox_y = stat[cv2.CC_STAT_TOP]
    bbox_w = stat[cv2.CC_STAT_WIDTH]
    bbox_h = stat[cv2.CC_STAT_HEIGHT]
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
    cv2.imshow(frame_label, cloned_frame)
    cv2.imshow("%s Mask" % frame_label, componentMask)
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


def run_with_stats(keystroke, font_size):
    im = keystroke.kisolation_frame
    im = enhance_image(im)
    im = np.array(im)

    # apply connected component analysis to the thresholded image
    cca_result = cv2.connectedComponentsWithStats(im, CONNECTIVITY, cv2.CV_32S)
    # display_regions(im, font_size, result)
    return the_algorithm(im, cca_result)
