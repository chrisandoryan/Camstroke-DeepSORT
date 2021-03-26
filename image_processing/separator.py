import cv2
from PIL import Image, ImageOps
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils
from helpers import constants
from helpers.utils import print_full
from skimage import io, morphology, filters
import matplotlib.pyplot as plt
from skan import draw
from operator import itemgetter

from helpers.image import display, save_image, pad_image

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
    # skeletonization using skimage
    # binary = np.array(im)
    # binary[binary == 255] = 1
    # sk_im = morphology.skeletonize(binary)
    # sk_im = sk_im > filters.threshold_otsu(sk_im)
    # sk_im = sk_im.astype('int8')
    # sk_im = cv2.normalize(sk_im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # display(sk_im, "sk")

    # skeletonization using opencv-contrib
    cv_im = cv2.ximgproc.thinning(im)
    # display(cv_im, "cv")

    return cv_im

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

def find_branch_point_coordinates(skeleton_im):
    w, h = skeleton_im.shape
    # print("W, H", w, h)

    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(skeleton_im)

    # Initialize empty list of co-ordinates
    skel_coords = []

    # For each non-zero pixels (non-zero means WHITE)
    for (r,c) in zip(rows,cols):
        # print(r, c)
        # Prevent neighbor checking beyond the border
        c_left = c - 1 if c != 0 else c
        c_right = c + 1 if c != h - 1 else c
        r_top = r - 1 if r != 0 else r
        r_bottom = r + 1 if r != w - 1 else r

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
    
    # get the top n rightmost x coordinate
    if len(skel_coords) > 0:
        rightmost_coord = max(skel_coords, key=itemgetter(0))
        # rightmost_coord = sorted(skel_coords, lambda tup: tup[0])
        return rightmost_coord
    
    return (None, None)

# https://stackoverflow.com/questions/53481596/python-image-finding-largest-branch-from-image-skeleton
def prune_branches(skeleton_im):
    kernel = np.ones((5, 3), np.uint8)
    pruned_im = cv2.erode(skeleton_im, kernel, iterations=3)
    return pruned_im

# this function is necessary to separate a character that intersect/overlap with the cursor (or another character), so that the captured timings can be more effective.
# https://stackoverflow.com/questions/14211413/segmentation-for-connected-characters
def solve_overlapping(candidate):
    crop_padding = 8

    overlap_im = candidate['mask']
    bbox_x = candidate['coord']['x'] - crop_padding
    bbox_y = candidate['coord']['y'] - crop_padding
    # bbox_w = candidate['shape']['w']
    # bbox_h = candidate['shape']['h']

    # crop the image according to region's bounding box
    overlap_im = overlap_im[bbox_y:,bbox_x:]
    overlap_im = pad_image(overlap_im, crop_padding)
    im_h, im_w = overlap_im.shape

    # perform basic image enhancement
    overlap_im = cv2.threshold(cv2.medianBlur(overlap_im, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # skeletonize the frame
    skeleton_im = skeletonization(overlap_im)
    # display(skeleton_im)

    # returned branch point coordinates will always be the rightmost coordinates
    bp_coord = find_branch_point_coordinates(skeleton_im)
    # print(bp_coord)

    if all(bp_coord):
        # create a mask until a few pixels before the intersection (bp) coordinate, as tall as the image
        bp_x, bp_y = bp_coord
        x_start, y_start = (0, 0)
        x_end, y_end = (bp_x - constants.INTERSECTION_PADDING, im_h)

        mask = np.zeros(overlap_im.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
        # display(mask, "The Mask")

        # erase the cursor object from the image based on intersection coordinate
        clean_im = cv2.bitwise_and(skeleton_im, skeleton_im, mask=mask)
        # clean_im_also = cv2.bitwise_and(overlap_im, overlap_im, mask=mask)

        color = (255, 0, 0)
        radius = 3
        thickness = 2
        separation_im = cv2.circle(skeleton_im, bp_coord, radius, color, thickness)
        # display the cutoff coordinate
        # display(im, "Junction Coordinate")

        # dilate the frame (de-skeletonize)
        kernel = np.ones((3, 3), np.uint8)
        dilated_im = cv2.dilate(clean_im, kernel, iterations=5)

        # TODO: prune small branches from the skeleton
        pruned_im = prune_branches(dilated_im)
        # display(pruned_im, "Test")

        # perform more image enhancement
        final_im = cv2.GaussianBlur(pruned_im, (21, 21), 0)
        final_im = cv2.morphologyEx(final_im, cv2.MORPH_OPEN, kernel)
        final_im = cv2.threshold(cv2.medianBlur(dilated_im, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        final_im = cv2.morphologyEx(dilated_im, cv2.MORPH_CLOSE, kernel)
        # display(final_im, "Final Result")

        return final_im, separation_im

    return None    
