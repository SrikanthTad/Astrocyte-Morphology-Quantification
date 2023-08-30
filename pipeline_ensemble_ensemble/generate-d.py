import os
import argparse
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from torchvision import transforms
from skimage.transform import resize

from data_loading import BasicDataset

from config.test_config import test_cfg
from dataloader.coco_dataset import coco
from utils.draw_box_utils import draw_box
from utils.train_utils import create_model
from segmentation import (
    get_segmentation,
    get_colored_mask,
    get_segmentation_nuclei,
    find_intersection,
    get_colored_mask_nuclei,
    AstrocyteTypes

)


from branch import find_branches, paint_branches

from torchvision.transforms.functional import crop

def test():
    """
    This function loads the object detection model + segnmentation model, plot the predictions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load the object detection model
    model_od = create_model(num_classes=test_cfg.num_classes)
    weights = test_cfg.model_weights
    checkpoint = torch.load(weights, map_location=device)
    model_od.load_state_dict(checkpoint['model'])
    model_od = model_od.to(device) #push to device
    # read class_indict
    data_transform = transforms.Compose([transforms.ToTensor()])

    # load the image
    original_img = Image.open(test_cfg.image_path)
    print("image shape: ", original_img.size)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)
    img_size = img.size()
    np.set_printoptions(threshold=np.inf)
    model_od.eval()
    with torch.no_grad():
        predictions = model_od(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        print("boxes: ", predict_boxes)
        print("classes: ", predict_classes)
        print("scores: ", predict_scores)


        if len(predict_boxes) == 0:
            print("No target detected!")

        # create to black frames, 1 for image where all branches have the same color, one for image
        # with separated branches
        blackFrame_1 = np.zeros([img_size[2], img_size[3], 3],dtype=np.uint8)
        blackFrame_2 = blackFrame_1.copy()


        cell_surface = 0
        primary_branches = 0
        nuclei_surface = 0
        number_of_nuclei = 0
        
        for i in range(len(predict_boxes)):
            left, top, right, bottom = [int(predict_boxes[i][j]) for j in range(4)]
            img = original_img.crop([left, top, right, bottom])

            # run the 2 segmentation models
            mask_astrocyte = get_segmentation(img)
            mask_nuclei = get_segmentation_nuclei(img)
           
            
            cell_surface = np.sum(mask_astrocyte[:, :, 0] != 0)
            nuclei_surface = np.sum(mask_nuclei[:, :, 0] != 0)

            # Draw Mask (BODY vs BRANCH)
            colored_astrocyte = get_colored_mask(mask_astrocyte)
            colored_nuclei = get_colored_mask_nuclei(mask_nuclei)
            

            # find the nuclues of the biggest intersection with the astrocyte's body
            biggest_intersection_nucleus, biggest_nucleus, n_nuclei  = find_intersection(
                mask_astrocyte, mask_nuclei
            )
            number_of_nuclei = n_nuclei

            if biggest_intersection_nucleus is not None:
                # give the interesection the color red:
                colored_astrocyte[biggest_intersection_nucleus] = [255, 0, 0]
                
                # paint the whole nucleus on the background. This line can be removed
                colored_astrocyte[(mask_astrocyte[:, :, 0]  == 0) & (biggest_nucleus == 1)] = [255, 128, 0]

            # find all the branches
            branches = find_branches(mask_astrocyte)

            primary_branches = len(branches)
            # color different branch by different colors
            colored_astrocyte = paint_branches(colored_astrocyte, branches)


            # put the box to the black frames
            blackFrame_1[top: bottom, left:right] = colored_astrocyte
            blackFrame_2[top: bottom, left:right] = colored_nuclei

        
        # save to file
        blackFrame_1 = blackFrame_1[:, :, ::-1]
        blackFrame_2 = blackFrame_2[:, :, ::-1]

        cv2.imwrite("pipeline_output.jpg", blackFrame_1)
        cv2.imwrite("colored_nuclei.jpg", blackFrame_2)

        print("-"*30 + "\n\n")
        print("Cell Surface: ", cell_surface)
        print("Primary branches: ", primary_branches)
        print("Nucleii Surface: ", nuclei_surface)
        print("Number of Nucleii: ", number_of_nuclei)
        print("Number of Astrocytes: ", len(predict_boxes))
        print("Saved to files pipeline_output.jpg and colored_nuclei.jpg")

if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    os.environ["CUDA_VISIBLE_DEVICES"] = test_cfg.gpu_id
    test()