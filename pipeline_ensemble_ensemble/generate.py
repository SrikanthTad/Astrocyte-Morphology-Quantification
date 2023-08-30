import os
import argparse
import logging

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
import segmentation_models_pytorch as smp
from utils1 import plot_img_and_mask

from config.test_config import test_cfg
from dataloader.coco_dataset import coco
from utils.draw_box_utils import draw_box
from utils.train_utils import create_model

from torchvision.transforms.functional import crop

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    h, w = np.asarray(full_img).shape
    #print(h, w)
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        #if net.n_classes > 1:
        #print(output.size())
        probs = F.softmax(output, dim=1)[0]
        #print(probs)
        #else:
        #    probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

        res = F.one_hot(full_mask.argmax(dim=0), 2).numpy()
        #print(res.shape)
        res = Image.fromarray(np.uint8(res))
        res = res.resize((w, h), resample=Image.NEAREST)
        res = torch.from_numpy(np.asarray(res))
        #print(res.size())
        #print(res.permute(2, 0, 1).numpy().shape)
        return res.permute(2, 0, 1).numpy()

def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
        #return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))



def test():
    #Load the object detection model
    model_od = create_model(num_classes=test_cfg.num_classes)

    model_od.cuda()
    weights = test_cfg.model_weights

    checkpoint = torch.load(weights, map_location='cuda')
    model_od.load_state_dict(checkpoint['model'])

    #Load the segmentation model
    model_ft = smp.Unet(
    encoder_name="mobilenet_v2", #"resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    activation="softmax"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ft.to(device=device)
    model_ft.load_state_dict(torch.load("./MODEL.pth", map_location=device))

    # read class_indict
    data_transform = transforms.Compose([transforms.ToTensor()])
    test_data_set = coco(test_cfg.data_root_dir, 'test', '2017', data_transform)
    category_index = test_data_set.class_to_coco_cat_id

    index_category = dict(zip(category_index.values(), category_index.keys()))

    original_img = Image.open(test_cfg.image_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)
    img_size = img.size()
    #print(img_size[2])
    #print(img_size[3])

    np.set_printoptions(threshold=np.inf)

    colormap = np.array([[0.82, 0.1, 0.26], #Alizaron
            [0.0, 1.0, 0.0], #Electric Green
            [1.0, 0.75, 0.0], #Amber
            [0.0, 0.0, 1.0], #Blue
            [0.13, 0.67, 0.8], #Ball Blue
            [0.1, 0.44, 0.37], #Bitter Sweet
            [0.47, 0.27, 0.23], #Bole
            [1.0, 0.0, 0.5], #Bright Pink
            [0.56, 0.0, 1.0], #Electric voilet
            [1.0, 1.0, 0.0], #Electric Yellow
            [0.1, 0.1, 0.44], #Midnight Blue
            [0.16, 0.67, 0.53], #Jungle Green
            [0.83, 0.83, 0.83], #Light Gray
            [1.0, 1.0, 0.88], #Light Yellow
            [0.5, 0.5, 0.0], #Olive
            [0.0, 1.0, 0.94], #Turquoise Blue
            ]) # from https://latexcolor.com/
    #print(colormap.shape[0])

    model_od.eval()
    model_ft.eval()
    with torch.no_grad():
        predictions = model_od(img.cuda())[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("No target detected!")

        '''
        print(predict_boxes)
        print(predict_boxes[0])
        print(len(predict_boxes))
        print(len(predict_boxes[0]))
        print(predict_boxes[0])
        print(predict_boxes[1])
        '''


        blackFrame = np.zeros([img_size[2], img_size[3], 3],dtype=np.uint8)
        blackFrame = Image.fromarray(blackFrame)
        #plt.figure()
        #plt.imshow(blackFrame, interpolation='none')
        #plt.axis([0, img_size[3], img_size[2], 0])
        #plt.axis('off')
        #plt.show()
        original_img1 = blackFrame

        '''
        I = np.asarray(original_img)
        I = np.array([I, I, I]).transpose((1, 2, 0))
        original_img1 = Image.fromarray(np.uint8(I))
        draw = ImageDraw.Draw(original_img1)

        for i in range(len(predict_boxes)):
            xmin, ymin, xmax, ymax = predict_boxes[i]
            (left, right, top, bottom) = (xmin * 1 - 2, xmax * 1 + 2, ymin * 1 - 2, ymax * 1 + 2)
            #print(totuple((colormap[i % colormap.shape[0]] * 255).astype(np.uint8)))
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], 
              width = 2 , fill = totuple((colormap[i % colormap.shape[0]] * 255).astype(np.uint8)))
        '''

        plt.figure()
        plt.imshow(original_img1, interpolation='none')
        plt.axis([0, img_size[3], img_size[2], 0])
        plt.axis('off')

        for i in range(len(predict_boxes)):
            #img = crop(original_img, ymin = predict_boxes[i][1], xmin = predict_boxes[i][0], xmax = predict_boxes[i][2], ymax = predict_boxes[i][3])
            #print(original_img.size)
            img = original_img.crop((predict_boxes[i][0], predict_boxes[i][1], predict_boxes[i][2], predict_boxes[i][3]))
            #print(predict_boxes[i][0], predict_boxes[i][1], predict_boxes[i][2], predict_boxes[i][3])
            #print(img.size)
            #print(transforms.ToTensor()(img).size())
            mask = predict_img(net=model_ft,
                           full_img=img,
                           scale_factor= 1,
                           out_threshold= 0.5,
                           device=device)
            
            #print(mask.shape)
            #print("mask0:", mask[0])
            #print("mask1:", mask[1])
            #result = mask_to_image(mask)

            '''Draw Mask'''
            result = np.array([mask[1], mask[1], mask[1]])
            result = np.transpose(result, (1, 2, 0))
            result = result * ((colormap[i % colormap.shape[0]] * 255).astype(np.uint8))
            #print(result.shape)
            #print(result)
            plt.imshow(result, extent = [predict_boxes[i][0], predict_boxes[i][2], predict_boxes[i][3], predict_boxes[i][1]], interpolation='none', alpha=0.7)
            #result.save("./fig/"+str(i)+".jpg")
        
        plt.show()
        plt.savefig('/home/stadiset/ImageSegmentation/Astrocyte-CV/multiclass/Faster-RCNN/pipeline_output.jpg')
        '''
        draw_box(original_img1,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 index_category,
                 thresh=0.3,
                 line_thickness=3)

        plt.figure()
        plt.imshow(original_img1, vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
        '''


if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    os.environ["CUDA_VISIBLE_DEVICES"] = test_cfg.gpu_id
    test()
