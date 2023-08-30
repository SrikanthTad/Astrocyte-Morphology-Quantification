"""
To run segmentation.py alone do:

python segmentation.py /path/to/the/image
"""

import sys

import cv2
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import pytorch_lightning as pl

from config.test_config import test_cfg


SIZE = 192
N_INPUT_CHANNEL = 3
N_CLASSES = 3

class AstrocyteTypes:
    BACKGROUND = 0
    BRANCH = 1
    BODY = 2
    NUCLEUS =3

class NucleiTypes:
    BACKGROUND = 0
    NUCLEUS = 1
    BORDER = 2

ASTROCYTE_CLASS_MAP = {
    0: AstrocyteTypes.BACKGROUND,
    1: AstrocyteTypes.BRANCH,
    2: AstrocyteTypes.BODY,
    3: AstrocyteTypes.NUCLEUS
}

NUCLEI_CLASS_MAP = {
    0: NucleiTypes.BACKGROUND,
    1: NucleiTypes.NUCLEUS,
    2: NucleiTypes.BORDER
}

ASTROCYTE_CLASSES = len(set(ASTROCYTE_CLASS_MAP.values()))
NUCLEI_CLASSES = len(set(NUCLEI_CLASS_MAP.values()))

ASTROCYTE_COLOR_MAP = {
    AstrocyteTypes.BACKGROUND: (0, 0, 0),
    AstrocyteTypes.BRANCH: (255, 255, 255),
    AstrocyteTypes.BODY: (0, 255, 0),
    AstrocyteTypes.NUCLEUS: (0, 255, 255) #NEW ONE

}

NUCLEI_COLOR_MAP = {
    NucleiTypes.BACKGROUND: (0, 0, 0),
    NucleiTypes.NUCLEUS: (0, 0, 255),  #NEW ONE
    NucleiTypes.BORDER: (0, 255, 0) #NEW ONE


}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_colored_mask(processed_mask):
    """
    this function gets the masks produced by the segmentation function and create different
    objects (background, branch, body, nuclei) with different colors
    """
    h, w, _ = processed_mask.shape

    colored_mask = np.zeros((h, w, 3))
    mask = processed_mask[:, :, 0]
    colored_mask[mask == AstrocyteTypes.BACKGROUND] = ASTROCYTE_COLOR_MAP[AstrocyteTypes.BACKGROUND]
    colored_mask[mask == AstrocyteTypes.BRANCH] = ASTROCYTE_COLOR_MAP[AstrocyteTypes.BRANCH]
    colored_mask[mask == AstrocyteTypes.BODY] = ASTROCYTE_COLOR_MAP[AstrocyteTypes.BODY]
    colored_mask[mask == AstrocyteTypes.NUCLEUS] = ASTROCYTE_COLOR_MAP[AstrocyteTypes.NUCLEUS]
    colored_mask = colored_mask.astype(np.uint8)
    return colored_mask

def get_colored_mask_nuclei(processed_mask):
    """
    this function gets the masks produced by the segmentation function and create different
    objects (background, branch, body, nuclei) with different colors
    """

    h, w, _ = processed_mask.shape

    colored_mask = np.zeros((h, w, 3))
    mask = processed_mask[:, :, 0]
    colored_mask[mask == NucleiTypes.BACKGROUND] = NUCLEI_COLOR_MAP[NucleiTypes.BACKGROUND]
    colored_mask[mask == NucleiTypes.NUCLEUS] = NUCLEI_COLOR_MAP[NucleiTypes.NUCLEUS]
    colored_mask[mask == NucleiTypes.BORDER] = NUCLEI_COLOR_MAP[NucleiTypes.BORDER]
    colored_mask = colored_mask.astype(np.uint8)
    return colored_mask

preprocessing_fn = get_preprocessing_fn('resnet50', pretrained='imagenet')

class SegmentationCell(pl.LightningModule):
    def __init__(self, architecture="unet", encoder_name=""):
        super().__init__()

        # TODO: add more architecture if you like
        if architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=N_INPUT_CHANNEL,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=NUCLEI_CLASSES,                      # model output channels (number of classes in your dataset)
            )
        elif architecture == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=N_INPUT_CHANNEL,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=NUCLEI_CLASSES,                      # model output channels (number of classes in your dataset)
            )
        elif architecture == "manet":
            self.model = smp.MAnet(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=N_INPUT_CHANNEL,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=NUCLEI_CLASSES,                      # model output channels (number of classes in your dataset)
            )

    def forward(self, x):
        return self.model(x.float())




# load astrocyte model
astrocyte_models = [
    SegmentationCell.load_from_checkpoint(
        model_config["weight"],
        architecture=model_config["architecture"],
        encoder_name=model_config["encoder_name"]
    )
    for model_config in test_cfg.astrocyte_model_list[:3]
]

# load nuclei models (this can be 1 or 3 models)
nuclei_models = [
    SegmentationCell.load_from_checkpoint(
        model_config["weight"],
        architecture=model_config["architecture"],
        encoder_name=model_config["encoder_name"]
    )
    for model_config in test_cfg.nuclei_model_list[:3]
]


def ensemble_prediction(image_batch, *models):
    """
    This function is for doing prediction using ensemble models
    """
    with torch.no_grad():
        for model in models:
            model.eval()

        logits = [model(image_batch) for model in models]
        logits = [torch.nn.functional.softmax(logit, dim=1) for logit in logits]
        logits = np.concatenate([logit[:, :, np.newaxis, :, :] for logit in logits], axis=2)

        logits = np.mean(logits, axis=2)

    return logits


def get_segmentation(image):
    """
    Get the cropped of the cell, predict the mask and return the mask in the original shape
    """
    image = np.array(image)
    original_h, original_w, _ = image.shape

    image = cv2.resize(image, (192, 192))       # (h, w, 3)

    image = preprocessing_fn(image)
    image = image.transpose(2, 0, 1)            # (c, h, w)
    image = np.expand_dims(image, 0)            # (n, c , h, w)
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        #output = astrocyte_segmentation_model.model(image).numpy()
        output = ensemble_prediction(image, *astrocyte_models)
        output = output[0].transpose(1, 2, 0).argmax(axis=-1, keepdims=True).astype(np.uint8)
        output = np.expand_dims(cv2.resize(output, (original_w, original_h)), axis=-1)

    return output

def get_segmentation_nuclei(image):
    """
    Get the cropped of the cell, predict the mask and return the mask in the original shape
    """
    # we only need to get 1 channel for the nuclei
    image = np.array(image)[:, :, 0:1]
    image = np.repeat(image, 3, axis=-1)

    original_h, original_w, _ = image.shape

    image = cv2.resize(image, (192, 192))

    image = preprocessing_fn(image)

    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        #output = nuclei_segmentation_model.model(image).numpy()
        output = ensemble_prediction(image, *nuclei_models)
        output = output[0].transpose(1, 2, 0).argmax(axis=-1, keepdims=True).astype(np.uint8)
        output = np.expand_dims(cv2.resize(output, (original_w, original_h)), axis=-1)

    return output

def find_nuclei(mask):
    """
    find a list of nuclei given the mask using Breadth First Search
    """
    mask = mask[:, :, 0]
    h, w = mask.shape
    checked_pixels = set([])
    cells = []
    for i in range(h):
        for j in range(w):
            pixel_id = i*10000 + j
            if mask[i, j] != NucleiTypes.NUCLEUS or  pixel_id in checked_pixels:
                continue

            cell = np.zeros((h, w))
            cell[i, j] = 1
            checked_pixels.add(pixel_id)

            # do BREADTH FIRST SEARCH and and find all the neightbor having the same color
            # first layer only has 1 pixel
            neighbors = [[[i, j]]]
            found_other_pixel = False

            while True:
                new_layers = []
                # loop through all pixcel in last layer
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

                            # Only keep the neighbors that are nucleus
                            if mask[tmp_i + dh, tmp_j + dw] == NucleiTypes.NUCLEUS:
                                new_layers.append([y, x])
                                cell[y, x] = 1
                                checked_pixels.add(pixel_id)

                # if no more pixel is found for the new layer, exit the while loop
                if len(new_layers) == 0:
                    break

                neighbors.append(new_layers)

            # ignore cell that is too small
            if np.sum(cell) > 30:
                cells.append(cell)
    return cells

def find_nuclei_intersection(astrocyte_mask, nuclei_list):
    """
    This function find the biggest intersection of nucleus and the astrocyte's body
    """

    # enlarge the nuclei a little bit to account for the borders of the nuclei
    kernel = np.ones((3, 3), np.uint8)
    nuclei_list = [cv2.dilate(image*255, kernel, iterations=1) for image in nuclei_list]
    nuclei_list = [ (image > 0) + 0 for image in nuclei_list]

    astrocyte_mask = astrocyte_mask[:, :, 0]

    intersections = [
        (
            (astrocyte_mask == AstrocyteTypes.BODY) | (astrocyte_mask == AstrocyteTypes.NUCLEUS)
            #(astrocyte_mask == AstrocyteTypes.BODY)        # use this if you model doesn't have nuclues
        ) & (
            nucleus_mask == 1
        )
        for nucleus_mask in nuclei_list
    ]

    # this is when there is no nucleus on the image
    if len(intersections) == 0:
        return None, None

    intersection_areas = [np.sum(intersection) for intersection in intersections]
    # this is when there are nuclei, but no intersection
    print("Size of each intersection: ", intersection_areas)
    if sum(intersection_areas) == 0:
        return None, None


    max_index = np.argmax(intersection_areas)
    return intersections[max_index], nuclei_list[max_index]


def find_intersection(mask_astrocyte, mask_nuclei):
    """
    Given the mask_astrocyte, mask_nuclei, return:
        (
            the biggest intersection region,
            the nuclues with biggest intersection,
            total number of nucleii on mask_nuclei
        )
          
    """
    nuclei_list =  find_nuclei(mask_nuclei)

    biggest_intersection_nucleus, biggest_nucleus = find_nuclei_intersection(mask_astrocyte, nuclei_list)
    return biggest_intersection_nucleus, biggest_nucleus, len(nuclei_list)