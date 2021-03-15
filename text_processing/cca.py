import cv2
from PIL import Image, ImageOps
import numpy as np

"""
Script that runs connected component labelling for isolating characters from image
"""

CONNECTIVITY = 8

def isolate_components(im, output):
    (numLabels, labels, stats, centroids) = output
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    # loop over the number of unique connected component labels
    # skip the first result since it usually is the background being isolated
    for i in range(1, numLabels):
        # extract the connected component statistics and centroid for
        # the current label
        bbox_x = stats[i, cv2.CC_STAT_LEFT]
        bbox_y = stats[i, cv2.CC_STAT_TOP]
        bbox_w = stats[i, cv2.CC_STAT_WIDTH]
        bbox_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (bbox_cX, bbox_cY) = centroids[i]

        # clone our original image (so we can draw on it) and then draw
        # a bounding box surrounding the connected component along with
        # a circle corresponding to the centroid
        cloned_frame = im.copy()
        cv2.rectangle(cloned_frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 3)
        cv2.circle(cloned_frame, (int(bbox_cX), int(bbox_cY)), 4, (0, 0, 255), -1)

        # construct a mask for the current connected component by
        # finding a pixels in the labels array that have the current
        # connected component ID
        componentMask = (labels == i).astype("uint8") * 255
        
        # show our output image and connected component mask
        cv2.imshow("Output", cloned_frame)
        # cv2.imshow("Connected Component", componentMask)
        cv2.waitKey(0)

def run_with_stats(im):
    im = np.array(im)

    # apply connected component analysis to the thresholded image
    result = cv2.connectedComponentsWithStats(im, CONNECTIVITY, cv2.CV_32S)
    isolate_components(im, result)
