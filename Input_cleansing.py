import os
import argparse
import time
import shutil
import math
import numpy as np
import torch
import shutup
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import numbers


from data_reader import CUEING_reader
from torchvision.utils import save_image
from functions import bb_mapping

parser = argparse.ArgumentParser(description='Cleansing')
parser.add_argument('--imgdir', metavar='DIR', help='path to images folder')
parser.add_argument('--gazemapsimg', metavar='DIR', help='path to gaze map images folder')
parser.add_argument('--grid', default='', type=str, metavar='PATH', help='path to txt with grid entries for training images')
parser.add_argument('--yolo5bb', metavar='DIR', help='path to folder of yolo5 bounding box txt files')
parser.add_argument('--height', default=16, type=int, metavar='N',
                    help='height of image')
parser.add_argument('--width', default=16, type=int, metavar='N',
                    help='width of image ')

parser.add_argument('--savedir', metavar='DIR', help='path to save cleansed image')



def mask_input():

    args = parser.parse_args()

    dataset = CUEING_reader('mask', 'training', args.grid, args.imgdir, 0.1 ,args.gazemapsimg)



    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16, shuffle=True,
        num_workers=2, pin_memory=True)

    for i, (input, img_name) in enumerate(tqdm(loader)):
        # print(input.shape)
        for j in range(len(img_name)):
            img_names = img_name[j]
            inputs = input[j]

            filename = os.path.join(args.yolo5bb, img_names + ".txt")
            print(filename)
            if os.path.exists(filename):
                # print(11111)

                mask_img = torch.zeros(3,args.height,args.width)

                with open(filename) as f:

                    for linestring in f:

                        line = linestring.split()

                        width = float(line[3])
                        height = float(line[4])
                        x_center = float(line[1])
                        y_center = float(line[2])

                        x_min, x_max, y_min, y_max = bb_mapping(x_center, y_center, width, height,img_width = args.width, img_height =args.height)

                        mask_img[:, y_min:y_max+1, x_min:x_max+1] = inputs[:, y_min:y_max+1, x_min:x_max+1]





                    save_image(mask_img, args.savedir + img_names + '.jpg')



if __name__ == "__main__":

    mask_input()
