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
    
    # show the output image
    display(im)
    return im

def find_branch_points(skeletion_im):
    branch_data = summarize(Skeleton(skeletion_im))

    # to draw skeleton network
    # pixel_graph, coordinates, degrees = skeleton_to_csgraph(skeletion_im)
    # fig, axes = plt.subplots(1, 2)
    # draw.overlay_skeleton_networkx(pixel_graph, coordinates, image=skeletion_im, axis=axes[0])
    return branch_data

# this function is necessary to separate a character that intersect/overlap with the cursor (or another character), so that the captured timings can be more effective.
# https://stackoverflow.com/questions/14211413/segmentation-for-connected-characters
def solve_overlapping(overlap_im):
    imw, imh = overlap_im.shape

    # skeletonize the frame
    im = skeletonization(overlap_im)
    display(im)

    # find branch point
    """
    Branch Type: 
    1: endpoint-to-endpoint (isolated branch)
    2: junction-to-endpoint
    3: junction-to-junction
    4: isolated cycle
    """
    bp = find_branch_points(im)
    print_full(bp)

    # dilate the frame (deskeletonize)
    kernel = np.ones((1, 1), np.uint8)
    

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
