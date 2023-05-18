import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import cv2
from utils import *
import torchvision
from PIL import Image

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def img_id(self):
        return (self._data[0]) # image index starts with 1

    @ property
    def  grids(self):
        grid=[]
        for item in self._data[1:]:
            grid.append(float(item))
        return grid


class CUEING_reader(Dataset):
    """
    BDDA feature class.
    """
    def __init__(self, task, subset, file, img_path, threshold, gazemap_path):
        """
        Args:

        """
        self.task = task

        self.subset = subset
        self.file = file
        self.img_path = img_path
        self.gazemap_path = gazemap_path
        self.threshold = threshold

        self._parse_list()


        self.transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize([36, 64]),
                torchvision.transforms.ToTensor()])
        self.transform_2 = torchvision.transforms.Compose(
                [torchvision.transforms.Resize([720,1280]),
                 torchvision.transforms.ToTensor()])
        self.transform_3 = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

    def _parse_list(self):
        self.img_list = []
        tmp = [x.strip().split(',') for x in open(self.file)]
        img_list = [VideoRecord(item) for item in tmp]



        for item in img_list:
            img_name = item.img_id.split('.')[0]
            im_name = img_name + ".jpg"
            grid = item.grids
            img_path = os.path.join(self.img_path, im_name)


            if os.path.exists(img_path) and not all(math.isnan(y) for y in grid):

                self.img_list.append(item)

        print('video number in %s: %d'%(self.subset,(len(self.img_list))))


    def _normalizeData(self, data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """
        """

        record = self.img_list[index]
        img = record.img_id
        img_name = record.img_id.split('.')[0]

        img = Image.open(f'{self.img_path}/' + img).convert('RGB')
        if self.task != 'mask':
            img = self.transform_2(img)
        else:
            img = self.transform_3(img)



        grid = np.array(record.grids)
        grid[grid>self.threshold] = 1.0
        grid[grid<=self.threshold] = 0.0
        grid = grid.astype(np.float32)



        # if self.task == 'main':

            # img = Image.open(f'{self.dataset}/{self.subset}/camera_images/' + img).convert('RGB')
        # else:
        #     img = Image.open(f'{self.dataset}/{self.subset}/camera_images/' + img).convert('RGB')




        name = record.img_id.split('_')
        gaze_file = name[0] + '_pure_hm_' + name[1]


        if self.task =='denoise':
            gaze_gt = Image.open(os.path.join(self.gazemap_path, gaze_file))
            gaze_gt = self.transform_3(gaze_gt)
        else:
            gaze_gt = Image.open(os.path.join(self.gazemap_path, gaze_file)).convert('L').crop(
                (0, 96, 1024, 672))  # left,top,right,bottom
            gaze_gt = self.transform(gaze_gt)
        gaze_gt = self._normalizeData(gaze_gt)




        if self.task == 'speed':
            return img, img_name

        if self.task =='mask':
            return img, img_name

        if self.task == 'denoise':
            return gaze_gt, img_name
        if self.task == 'main' and self.subset != 'testing':
             return img, grid, gaze_gt, img_name
        if self.task == 'main' and self.subset == 'testing':
             return img, gaze_gt, img_name