import os, sys
import cv2
import json
import numpy as np
from collections import Counter

from PIL import Image
import networkx as nx
from segmentation import get_segmentation, get_colored_mask, AstrocyteTypes


# add more colors if you have many more branches
BRANCH_COLORS = [
    (175, 242, 51),
    (245, 122, 182),
    (245, 66, 66),
    (44, 125, 27),
    (0, 255, 145),
    (255, 0, 51),
    (0, 213, 255),
    (227, 158, 118),
    (0, 119, 255),
    (242, 237, 78),
    (174, 0, 255),
    (176, 4, 156)
]

MIN_PIXEL_PER_BRANCH = 7

def paint_branches(colored_astrocyte, branches):
    """
    This function paints different branch by different colors
    """

    for i, branch in enumerate(branches):
        colored_astrocyte[branch == 1] = BRANCH_COLORS[i]

    return colored_astrocyte

def find_branches(mask):
    """
    find a list of branches given the masks
    """
    mask = mask[:, :, 0]
    h, w = mask.shape
    checked_pixels = set([])
    branches = []
    for i in range(h):
        for j in range(w):
            pixel_id = i*10000 + j
            # Step 1: look for a pixel inside a branch
            if mask[i, j] != AstrocyteTypes.BRANCH or pixel_id in checked_pixels:
                continue

            branch = np.zeros((h, w))
            branch[i, j] = 1
            checked_pixels.add(pixel_id)

            # Step 2: do BREADTH FIRST SEARCH, starting from the pixel on step 1
            # and find all the neightbors and assign the same having the same color
            # first layer only has 1 pixel
            neighbors = [[[i, j]]]
            found_other_pixel = False

            while True:
                new_layers = []
                # loop through all pixel in previous layer
                for last_point in neighbors[-1]:

                    # find all their neighbor and add add to the new_layers
                    tmp_i, tmp_j = last_point
                    for dh in [-1, 0, 1]:
                        for dw in [-1, 0, 1]:
                            if dh == 0 and dw == 0 :
                                continue

                            y, x = tmp_i + dh, tmp_j + dw
                            if y < 0 or y >= h or x < 0 or x >= w:
                                continue
                            pixel_id = (y)*10000 + x
                            if pixel_id in checked_pixels:
                                continue

                            # A neighbor must be also a branch cell
                            if mask[tmp_i + dh, tmp_j + dw] == AstrocyteTypes.BRANCH:
                                new_layers.append([y, x])
                                branch[y, x] = 1
                                checked_pixels.add(pixel_id)

                # if no more pixel is found for the new layer, exit the while loop
                if len(new_layers) == 0:
                    break

                
                neighbors.append(new_layers)

            # check if the area of the branch is bigger than MIN_PIXEL_PER_BRANCH pixels
            if np.sum(branch) > MIN_PIXEL_PER_BRANCH:
                branches.append(branch)

    # keep the top 10 biggest branches
    branch_sizes = np.array([np.sum(branch) for branch in branches])
    branches = [branches[i] for i in np.argsort(-branch_sizes)[:len(BRANCH_COLORS)]]
    
    return branches
