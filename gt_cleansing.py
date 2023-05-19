import os
import math
import torch
from torch.utils.data import DataLoader
from data_reader import CUEING_reader
from torchvision.utils import save_image
from functions import bb_mapping
from tqdm import tqdm
import numpy as np
import argparse

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



def noise_filter():

    args = parser.parse_args()


    dataset = CUEING_reader('denoise',  'training', args.grid, args.imgdir, 0.1, args.gazemapsimg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=2, pin_memory=True)

    for i, (gaze_gt, img_name) in enumerate(tqdm(loader)):

        for j in range(len(img_name)):
            im_name = img_name[j]
            gt_img = gaze_gt[j]

            filtered_gaze = torch.zeros([3,args.height,args.width])

            filename = os.path.join(args.yolo5bb, im_name + ".txt")

            if os.path.exists(filename):
                with open(filename) as f:
                    for linestring in f:

                        line = linestring.split()

                        width = float(line[3])
                        height = float(line[4])
                        x_center = float(line[1])
                        y_center = float(line[2])

                        x_min, x_max, y_min, y_max = bb_mapping(x_center, y_center, width, height,img_width = args.width, img_height = args.height)

                        filtered_gaze[:, y_min:y_max + 1, x_min:x_max + 1] = gt_img[:, y_min:y_max + 1, x_min:x_max + 1]


            gaze_img_name = im_name.split('_')[0]+'_pure_hm_'+im_name.split('_')[1]

            gaze_map_filtered = os.path.join(args.savedir,gaze_img_name+'.jpg')
            save_image(filtered_gaze, gaze_map_filtered,'jpeg')







if __name__ == "__main__":


  noise_filter()
