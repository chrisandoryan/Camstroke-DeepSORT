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
import itertools

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

    # Functions to generate kernels of curve intersection

    def generate_nonadjacent_combination(input_list, take_n):
        """ 
        It generates combinations of m taken n at a time where there is no adjacent n.
        INPUT:
            input_list = (iterable) List of elements you want to extract the combination 
            take_n =     (integer) Number of elements that you are going to take at a time in
                         each combination
        OUTPUT:
            all_comb =   (np.array) with all the combinations
        """
        all_comb = []
        for comb in itertools.combinations(input_list, take_n):
            comb = np.array(comb)
            d = np.diff(comb)
            fd = np.diff(np.flip(comb))
            if len(d[d == 1]) == 0 and comb[-1] - comb[0] != 7:
                all_comb.append(comb)
                # print(comb)
        return all_comb

    def populate_intersection_kernel(combinations):
        """
        Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
        a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And 
        generates a kernel that represents a line intersection, where the center 
        pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
        INPUT:
            combinations = (np.array) matrix where every row is a vector of combinations
        OUTPUT:
            kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
        """
        n = len(combinations[0])
        template = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")
        match = [(0, 1), (0, 2), (1, 2), (2, 2),
                 (2, 1), (2, 0), (1, 0), (0, 0)]
        kernels = []
        for n in combinations:
            tmp = np.copy(template)
            for m in n:
                tmp[match[m][0], match[m][1]] = 1
            kernels.append(tmp)
        return kernels

    def give_intersection_kernels():
        """
        Generates all the intersection kernels in a 9x9 matrix.
        INPUT:
            None
        OUTPUT:
            kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
        """
        input_list = np.arange(8)
        taken_n = [4, 3]
        kernels = []
        for taken in taken_n:
            comb = generate_nonadjacent_combination(input_list, taken)
            tmp_ker = populate_intersection_kernel(comb)
            kernels.extend(tmp_ker)
        return kernels

    # Find the curve intersections

    def find_line_intersection(input_image, show=0):
        """
        Applies morphologyEx with parameter HitsMiss to look for all the curve 
        intersection kernels generated with give_intersection_kernels() function.
        INPUT:
            input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
        OUTPUT:
            output_image = (np.array dtype=np.uint8) image where the nonzero pixels 
                           are the line intersection.
        """
        kernel = np.array(give_intersection_kernels())
        output_image = np.zeros(input_image.shape)
        for i in np.arange(len(kernel)):
            out = cv2.morphologyEx(
                input_image, cv2.MORPH_HITMISS, kernel[i, :, :])
            output_image = output_image + out
        if show == 1:
            show_image = np.reshape(np.repeat(
                input_image, 3, axis=1), (input_image.shape[0], input_image.shape[1], 3))*255
            show_image[:, :, 1] = show_image[:, :, 1] - output_image * 255
            show_image[:, :, 2] = show_image[:, :, 2] - output_image * 255
            plt.imshow(show_image)
        return output_image

    #  finding corners
    def find_endoflines(input_image, show=0):
        """
        """
        kernel_0 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, 1, -1]), dtype="int")

        kernel_1 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [1, -1, -1]), dtype="int")

        kernel_2 = np.array((
            [-1, -1, -1],
            [1, 1, -1],
            [-1, -1, -1]), dtype="int")

        kernel_3 = np.array((
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")

        kernel_4 = np.array((
            [-1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")

        kernel_5 = np.array((
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")

        kernel_6 = np.array((
            [-1, -1, -1],
            [-1, 1, 1],
            [-1, -1, -1]), dtype="int")

        kernel_7 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]), dtype="int")

        kernel = np.array((kernel_0, kernel_1, kernel_2,
                           kernel_3, kernel_4, kernel_5, kernel_6, kernel_7))
        output_image = np.zeros(input_image.shape)
        for i in np.arange(8):
            out = cv2.morphologyEx(
                input_image, cv2.MORPH_HITMISS, kernel[i, :, :])
            output_image = output_image + out

        if show == 1:
            show_image = np.reshape(np.repeat(
                input_image, 3, axis=1), (input_image.shape[0], input_image.shape[1], 3))*255
            show_image[:, :, 1] = show_image[:, :, 1] - output_image * 255
            show_image[:, :, 2] = show_image[:, :, 2] - output_image * 255
            plt.imshow(show_image)

        return output_image  # , np.where(output_image == 1)


def find_branch_point_coordinates(skeleton_im):
    w, h = skeleton_im.shape
    print("w", w)
    print("h", h)
    # Find row and column locations that are non-zero
    (rows, cols) = np.nonzero(skeleton_im)

    # Initialize empty list of co-ordinates
    skel_coords = []

    # For each non-zero pixel...
    for (r, c) in zip(rows, cols):
        # print("r, c", r, c)
        if r != h - 1 and c != w - 1:
            # Prevent neighbor checking beyond the border
            c_left = c - 1 if c != 0 else c
            c_right = c + 1 if c != w - 1 else c
            r_top = r - 1 if r != 0 else r
            r_bottom = r + 1 if r != h - 1 else r
            # Extract an 8-connected neighbourhood
            (col_neigh, row_neigh) = np.meshgrid(
                np.array([c_left, c, c_right]), np.array([r_top, r, r_bottom]))
            # Cast to int to index into image
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')
            # print(col_neigh, row_neigh)
            # Convert into a single 1D array and check for non-zero locations
            pix_neighbourhood = skeleton_im[row_neigh, col_neigh].ravel() != 0
            # If the number of non-zero locations equals 2, add this to
            # our list of co-ordinates
            if np.sum(pix_neighbourhood) == 3:
                skel_coords.append((c, r))

    # print("".join(["(" + str(r) + "," + str(c) + ")\n" for (r,c) in skel_coords]))
    return skel_coords

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

# Functions to generate kernels of curve intersection 
def generate_nonadjacent_combination(input_list,take_n):
    """ 
    It generates combinations of m taken n at a time where there is no adjacent n.
    INPUT:
        input_list = (iterable) List of elements you want to extract the combination 
        take_n =     (integer) Number of elements that you are going to take at a time in
                     each combination
    OUTPUT:
        all_comb =   (np.array) with all the combinations
    """
    all_comb = []
    for comb in itertools.combinations(input_list, take_n):
        comb = np.array(comb)
        d = np.diff(comb)
        fd = np.diff(np.flip(comb))
        if len(d[d==1]) == 0 and comb[-1] - comb[0] != 7:
            all_comb.append(comb)        
            # print(comb)
    return all_comb

def populate_intersection_kernel(combinations):
    """
    Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
    a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And 
    generates a kernel that represents a line intersection, where the center 
    pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
    INPUT:
        combinations = (np.array) matrix where every row is a vector of combinations
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    n = len(combinations[0])
    template = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")
    match = [(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(0,0)]
    kernels = []
    for n in combinations:
        tmp = np.copy(template)
        for m in n:
            tmp[match[m][0],match[m][1]] = 1
        kernels.append(tmp)
    return kernels

def give_intersection_kernels():
    """
    Generates all the intersection kernels in a 9x9 matrix.
    INPUT:
        None
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    input_list = np.arange(8)
    taken_n = [4,3]
    kernels = []
    for taken in taken_n:
        comb = generate_nonadjacent_combination(input_list,taken)
        tmp_ker = populate_intersection_kernel(comb)
        kernels.extend(tmp_ker)
    return kernels

# Find the curve intersections
def find_line_intersection(input_image, show=0):
    """
    Applies morphologyEx with parameter HitsMiss to look for all the curve 
    intersection kernels generated with give_intersection_kernels() function.
    INPUT:
        input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
    OUTPUT:
        output_image = (np.array dtype=np.uint8) image where the nonzero pixels 
                       are the line intersection.
    """
    kernel = np.array(give_intersection_kernels())
    output_image = np.zeros(input_image.shape)
    for i in np.arange(len(kernel)):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i,:,:])
        output_image = output_image + out
    return cv2.findNonZero(output_image)

#  finding corners
def find_endoflines(input_image, show=0):
    """
    """
    kernel_0 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, 1, -1]), dtype="int")
    kernel_1 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [1,-1, -1]), dtype="int")
    kernel_2 = np.array((
            [-1, -1, -1],
            [1, 1, -1],
            [-1,-1, -1]), dtype="int")
    kernel_3 = np.array((
            [1, -1, -1],
            [-1, 1, -1],
            [-1,-1, -1]), dtype="int")
    kernel_4 = np.array((
            [-1, 1, -1],
            [-1, 1, -1],
            [-1,-1, -1]), dtype="int")
    kernel_5 = np.array((
            [-1, -1, 1],
            [-1, 1, -1],
            [-1,-1, -1]), dtype="int")
    kernel_6 = np.array((
            [-1, -1, -1],
            [-1, 1, 1],
            [-1,-1, -1]), dtype="int")
    kernel_7 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1,-1, 1]), dtype="int")
    kernel = np.array((kernel_0,kernel_1,kernel_2,kernel_3,kernel_4,kernel_5,kernel_6, kernel_7))
    output_image = np.zeros(input_image.shape)
    for i in np.arange(8):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i,:,:])
        output_image = output_image + out
    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),(input_image.shape[0],input_image.shape[1],3))*255
        show_image[:,:,1] = show_image[:,:,1] -  output_image *255
        show_image[:,:,2] = show_image[:,:,2] -  output_image *255
        plt.imshow(show_image)    
    return output_image#, np.where(output_image == 1)

# this function is necessary to separate a character that intersect/overlap with the cursor (or another character), so that the captured timings can be more effective.
# https://stackoverflow.com/questions/14211413/segmentation-for-connected-characters
def solve_overlapping(overlap_im):
    imw, imh = overlap_im.shape

    # skeletonize the frame
    im = skeletonization(overlap_im)
    display(im)

    # find branch point
    # bp = find_branch_points(im)
    # print_full(bp)

    # bp_coords = find_branch_point_coordinates(im)
    # print(bp_coords)
    # color = (255, 0, 0)
    # radius = 10
    # thickness = 2
    # for coord in bp_coords:
    #     im = cv2.circle(im, coord, radius, color, thickness)
    #     im = cv2.circle(im, coord, radius, color, thickness)

    # eol_img = find_endoflines(im, 0)
    coords = find_line_intersection(im, 0)
    print(coords)

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
    im = cv2.threshold(cv2.bilateralFilter(im, 5, 75, 75), 0,
                       255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return im


def pad_image(image, target_size=100):
    padded_image = ImageOps.expand(image, target_size, 'black')
    return padded_image
