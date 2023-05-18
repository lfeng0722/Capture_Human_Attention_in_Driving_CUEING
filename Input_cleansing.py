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


def mask_input(yolo5bb,loader,width, height, save_dir):

    for i, (input, img_name) in enumerate(loader):
        print(input.shape)
        for j in range(len(img_name)):
            img_names = img_name[j]
            inputs = input[j]

            filename = os.path.join(yolo5bb, img_names + ".txt")
            if os.path.exists(filename):

                mask_img = torch.zeros(3,660,1584)

                with open(filename) as f:

                    for linestring in f:

                        line = linestring.split()

                        width = float(line[3])
                        height = float(line[4])
                        x_center = float(line[1])
                        y_center = float(line[2])

                        x_min, x_max, y_min, y_max = bb_mapping(x_center, y_center, width, height,img_width = width, img_height = height)

                        mask_img[:, y_min:y_max+1, x_min:x_max+1] = inputs[:, y_min:y_max+1, x_min:x_max+1]





                    save_image(mask_img, save_dir + img_names + '.jpg')



if __name__ == "__main__":

    subset = 'training'
    task = 'mask'


    grid = 'grids/grid1616/testing_dada_grid.txt'
    #location of your grid.txt

    gazemap = 'DADA/testing/gazemap_images'
    #location of your gazemap

    img = 'DADA/testing/camera_images'
    #location of your original image

    yolobb = f'yolo5_boundingboxes/testing_dada'
    #location of your bounding box

    width, height = 1584, 660
    #the size of your original input

    save_dir = ''
    #location to save your cleansed input


    dataset = CUEING_reader(task, subset, grid, img, 0.1 ,gazemap)



    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16, shuffle=True,
        num_workers=2, pin_memory=True)
    mask_input(yolobb,loader,width, height, save_dir)
