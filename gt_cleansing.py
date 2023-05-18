import os
import math
import torch
from torch.utils.data import DataLoader
from data_reader import CUEING_reader
from torchvision.utils import save_image
from functions import bb_mapping
from tqdm import tqdm
import numpy as np
def noise_filter(subset, data, yolo_bb, grid, img_dir, gaze_map):

    task = 'denoise'
    th = 0.1

    dataset = CUEING_reader(task, data, subset, grid, img_dir, th, gaze_map, False, 6)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=2, pin_memory=True)

    for i, (gaze_gt, img_name) in enumerate(tqdm(loader)):

        for j in range(len(img_name)):
            im_name = img_name[j]
            gt_img = gaze_gt[j]

            # print(gt_img.shape)
            filtered_gaze = torch.zeros([3,660,1584])

            filename = os.path.join(yolo_bb, im_name + ".txt")

            if os.path.exists(filename):
                with open(filename) as f:
                    for linestring in f:

                        line = linestring.split()

                        width = float(line[3])
                        height = float(line[4])
                        x_center = float(line[1])
                        y_center = float(line[2])

                        x_min, x_max, y_min, y_max = bb_mapping(x_center, y_center, width, height,img_width = 1584, img_height = 660)

                        filtered_gaze[:, y_min:y_max + 1, x_min:x_max + 1] = gt_img[:, y_min:y_max + 1, x_min:x_max + 1]
                        # print(torch.max(gt_img))
                        # filtered_gaze =gt_img
                        # print(torch.max(filtered_gaze))

            gaze_img_name = im_name.split('_')[0]+'_pure_hm_'+im_name.split('_')[1]

            gaze_map_filtered = os.path.join('masked_dada/testing/fgazemap_images',gaze_img_name+'.jpg')
            save_image(filtered_gaze, gaze_map_filtered,'jpeg')







if __name__ == "__main__":
    data = 'BDDA'
    subset='training'
    yolo_bb = 'yolo5_boundingboxes/testing_dada'
    grid = 'grids/grid1616/testing_dada_grid.txt'
    img_dir = 'DADA/testing/camera_images'
    gaze_map = 'DADA/testing/gazemap_images'

    a = noise_filter(subset,data,yolo_bb,grid , img_dir, gaze_map)
