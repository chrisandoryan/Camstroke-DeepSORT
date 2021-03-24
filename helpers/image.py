"""
Credit to:
https://stackoverflow.com/questions/28935983/preprocessing-image-for-tesseract-ocr-with-opencv
https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
"""
import cv2
from PIL import Image, ImageOps
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils
from helpers import constants
from helpers.utils import print_full
from skimage import io, morphology
from skan import skeleton_to_csgraph
from skan import Skeleton, summarize
import matplotlib.pyplot as plt
from skan import draw
from operator import itemgetter

def display(im, title="An Image"):
    cv2.imshow(title, im)
    cv2.waitKey(0)

def convexity_defects(im):
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 2, 1)
    cnt = contours[0]
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(im, start, end, [0, 255, 0], 2)
            cv2.circle(im, far, 5, [0, 0, 255], -1)
    return im

def auto_canny(im, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(im)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(im, lower, upper)

	# return the edged image
	return edged

def skeletonization(im):
    # im = morphology.skeletonize(im)
    im = cv2.ximgproc.thinning(im)
    return im

def perform_watershed(im):
    thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(im.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(im, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(im, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return im

def find_branch_points(skeleton_im):
    """
    Using Skan library
    Branch Type: 
    1: endpoint-to-endpoint (isolated branch)
    2: junction-to-endpoint
    3: junction-to-junction
    4: isolated cycle
    - junctions (where three or more skeleton branches meet)
    - endpoints (where a skeleton ends)
    - paths (pixels on the inside of a skeleton branch
    """
    branch_data = summarize(Skeleton(skeleton_im))
    # to draw skeleton network
    # pixel_graph, coordinates, degrees = skeleton_to_csgraph(skeleton_im)
    # fig, axes = plt.subplots(1, 1)
    # axes = draw.overlay_skeleton_networkx(pixel_graph, coordinates, axis=axes)
    # plt.show()
    return branch_data

def find_branch_point_coordinates(skeleton_im):
    w, h = skeleton_im.shape
    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(skeleton_im)

    # Initialize empty list of co-ordinates
    skel_coords = []

    # For each non-zero pixels (non-zero means WHITE)
    for (r,c) in zip(rows,cols):
        # Prevent neighbor checking beyond the border
        c_left = c - 1 if c != 0 else c
        c_right = c + 1 if c != w - 1 else c
        r_top = r - 1 if r != 0 else r
        r_bottom = r + 1 if r != h - 1 else r

        # Extract an 8-connected neighbourhood
        (col_neigh, row_neigh) = np.meshgrid(np.array([c_left, c, c_right]), np.array([r_top, r,r_bottom]))

        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')

        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = skeleton_im[row_neigh,col_neigh].ravel() != 0

        # print("PN", pix_neighbourhood)
        # print("Result", np.sum(pix_neighbourhood))

        # If the number of non-zero locations equals 2, add this to 
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) >= 4:
            skel_coords.append((c, r))
    
    # get the rightmost x coordinate
    if len(skel_coords) > 0:
        rightmost_coord = max(skel_coords, key=itemgetter(0))
        return rightmost_coord
    
    return None

# this function is necessary to separate a character that intersect/overlap with the cursor (or another character), so that the captured timings can be more effective.
# https://stackoverflow.com/questions/14211413/segmentation-for-connected-characters
def solve_overlapping(overlap_im):
    im_w, im_h = overlap_im.shape

    # skeletonize the frame
    skeleton_im = skeletonization(overlap_im)
    # display(skeleton_im)

    # find branch point
    # bp = find_branch_points(im)
    # print_full(bp)

    # for i, b in bp.iterrows():
    #     x0 = int(b['image-coord-src-0'])
    #     y0 = int(b['image-coord-dst-0'])
    #     x1 = int(b['image-coord-src-1'])
    #     y1 = int(b['image-coord-dst-1'])
    #     print(x0, y0)
    #     print(x1, y1)
        
    # display(im)

    # returned branch point coordinates will always be the rightmost coordinates
    bp_coord = find_branch_point_coordinates(skeleton_im)
    print(bp_coord)

    # create a mask until a few pixels before the intersection (bp) coordinate, as tall as the image
    bp_x, bp_y = bp_coord
    x_start, y_start = (0, 0)
    x_end, y_end = (bp_x - 5, im_h)

    mask = np.zeros(overlap_im.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
    # display(mask, "The Mask")

    # erase the cursor object from the image based on intersection coordinate
    clean_im = cv2.bitwise_and(skeleton_im, skeleton_im, mask=mask)
    clean_im_also = cv2.bitwise_and(overlap_im, overlap_im, mask=mask)

    # to display the coordinate
    # color = (255, 0, 0)
    # radius = 3
    # thickness = 2
    # im = cv2.circle(im, bp_coord, radius, color, thickness)
    # display(im, "Junction Coordinate")

    # dilate the frame (de-skeletonize)
    kernel = np.ones((5, 5), np.uint8)
    final_im = cv2.dilate(clean_im, kernel, iterations=3)

    # display(final_im, "Final Result")

    # can't be used because the crop is unoptimal (some of the cursor region still appear)
    # display(clean_im_also, "Final Result 2")
    return final_im
    

def enhance_image(im):
    # resize image
    im = cv2.resize(im, None, fx=constants.RESIZE_FACTOR,
                    fy=constants.RESIZE_FACTOR, interpolation=cv2.INTER_CUBIC)

    # pyramid mean shift filtering
    # im = cv2.pyrMeanShiftFiltering(im, 21, 51)

    # display(im)

    # convert image to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # canny edge detection
    # im = auto_canny(im)

    # to display the image
    # display(im)

    # automatic thresholding using Otsu's algorithm
    thres, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # im = convexity_defects(im)
    # display(im, "after convdefects")

    # applying dilation and erosion
    kernel = np.ones((1, 1), np.uint8)
    im = cv2.erode(im, kernel, iterations=4)
    im = cv2.dilate(im, kernel, iterations=1)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    # applying adaptive blur
    # im = cv2.adaptiveThreshold(cv2.bilateralFilter(im, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # im = cv2.adaptiveThreshold(cv2.medianBlur(im, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # applying normal blur
    # im = cv2.threshold(cv2.medianBlur(im, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # im = cv2.threshold(cv2.GaussianBlur(im, (1, 1), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    im = cv2.threshold(cv2.bilateralFilter(im, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return im


def pad_image(image, target_size=100):
    padded_image = ImageOps.expand(image, target_size, 'black')
    return padded_image
