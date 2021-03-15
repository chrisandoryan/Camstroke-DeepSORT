import cv2

CONNECTIVITY = 8

def run_with_stats():
    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    