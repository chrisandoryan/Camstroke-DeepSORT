import cv2
from PIL import Image, ImageOps
import numpy as np

"""
Performs connected component labelling (CCA/CCL) for isolating characters from image.
Algorithm:
1. perform CCA and isolate character regions
2. eliminate regions that are cut off (the bbox's x or y is intersected with the border of the image)
3. if there is two object with exact same X (indicates both aligned vertically), combine both (for i and j case)
4. the typed character is the rightmost isolated region in the frame (karena karakter terakhir pasti berada tepat sebelum kursor, dimana kursor berada di kanan)
"""

CONNECTIVITY = 8

def the_algorithm():

    return

def isolate_components(im, output):
    (numLabels, labels, stats, centroids) = output
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im_height, im_width, _ = im.shape

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

        print("W", bbox_w)
        print("H", bbox_h)
        print("Area", area)
        print("X", bbox_x)
        print("Y", bbox_y)

        # ensure the width, height, and area are all neither too small
        # nor too big
        # Filter 1: eliminate region that intersect with frame's border
        testBorder = bbox_x + bbox_w < im_width or bbox_x > im_width - bbox_w

        # Filter 2: eliminate region that has width smaller than 1/3 of calculated font width
        testX = True # testBorder or (bbox_w > 8 and bbox_w < 50)

        # Filter 3: eliminate region that has height bigger than calculated font height
        testY = True # bbox_h > 45 and bbox_h < 65

        # Filter 4: is the test required?
        testArea = True # area > 500 and area < 1500

        print("Test Border: ", testBorder)
        print("TestX", testX)
        print("TestY", testY)

        # ensure the connected component we are examining passes all
        # three tests
        if all((testBorder, testX, testY, testArea)):
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
            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)

def run_with_stats(im):
    im = np.array(im)

    # apply connected component analysis to the thresholded image
    result = cv2.connectedComponentsWithStats(im, CONNECTIVITY, cv2.CV_32S)
    isolate_components(im, result)
