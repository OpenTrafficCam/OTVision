"""
Utils for using iou tracker
"""

# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License, see
# https://github.com/bochinski/iou-tracker/blob/master/LICENSE
# for details.
# Written by Erik Bochinski
# ---------------------------------------------------------

from typing import Union

import numpy as np


# TODO: Remove if not needed
def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    overlapThresh: float,
    classes: Union[np.ndarray, None] = None,
) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    perform non-maximum suppression. based on Malisiewicz et al.
    Args:
        boxes (numpy.ndarray): boxes to process
        scores (numpy.ndarray): corresponding scores for each box
        overlapThresh (float): overlap threshold for boxes to merge
        classes (numpy.ndarray, optional): class ids for each box.

    Returns:
        (tuple): tuple containing:

        boxes (list): nms boxes
        scores (list): nms scores
        classes (list, optional): nms classes if specified
    """
    # # if there are no boxes, return an empty list
    # if len(boxes) == 0:
    #     return [], [], [] if classes else [], []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    if scores.dtype.kind == "i":
        scores = scores.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # score = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs,
            np.concatenate((np.array([last]), np.where(overlap > overlapThresh)[0])),
        )

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]


def iou(
    bbox1: Union[list[float], tuple[float, float, float, float]],
    bbox2: Union[list[float], tuple[float, float, float, float]],
) -> float:
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union
