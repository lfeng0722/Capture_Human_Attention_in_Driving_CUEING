import os
import argparse
import time
import math
import numpy as np
import torch
import shutup
from torch.utils.data import DataLoader
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from torch.nn import functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numbers


import cv2
from sklearn.metrics import f1_score,precision_score,recall_score, roc_curve, roc_auc_score



def grid2heatmap(grid, size, num_grid):
    """
    Rearrange and expand gridvector of size (gridheight*gridwidth) to size (576 x 1024) by duplicating values

    :param grid: output vector
    :param size: (H,W) of one expanded grid cell
    :param num_grids: (H,W) = grid dimension
    :param args: parser arguments
    :return: 2D grid of size (576 x 1024)
    """
    new_heatmap = torch.zeros(grid.size(0),size[0]*num_grid[0],size[1]*num_grid[1])
    for i, item in enumerate(grid):
        idx = torch.nonzero(item)
        if idx.nelement() == 0:
            print('Empty')
            continue
        for x in idx:
            test = new_heatmap[i,x//num_grid[1]*size[0]:(x//num_grid[1]+1)*size[0],x%num_grid[1]*size[1]:(x%num_grid[1]+1)*size[1]]
            new_heatmap[i,x//num_grid[1]*size[0]:(x//num_grid[1]+1)*size[0],x%num_grid[1]*size[1]:(x%num_grid[1]+1)*size[1]] = item[x]
    output = new_heatmap.unsqueeze(1).cuda()

    return output

def cc(s_map_all,gt_all):
	eps = 1e-07
	bs = s_map_all.size()[0]
	r = 0
	for i in range(0, bs):
		s_map = s_map_all[i,:,:,:].squeeze()
		gt = gt_all[i,:,:,:].squeeze()
		s_map_norm = (s_map - torch.mean(s_map))/(eps + torch.std(s_map))
		gt_norm = (gt - torch.mean(gt))/(eps + torch.std(gt))
		a = s_map_norm.cpu()
		b = gt_norm.cpu()
		r += torch.sum(a*b) / (torch.sqrt(torch.sum(a*a) * torch.sum(b*b))+eps)
	return r/bs

def kl(s_map_all, gt_all):
	dims = len(s_map_all.size())
	bs = s_map_all.size()[0]
	eps = torch.tensor(1e-07)
	kl = 0

	if dims > 3:
		for i in range(0, bs):
			s_map = s_map_all[i,:,:,:].squeeze()
			gt = gt_all[i,:,:,:].squeeze()
			s_map = s_map/(torch.sum(s_map)*1.0 + eps)
			gt = gt/(torch.sum(gt)*1.0 + eps)
			gt = gt.to('cpu')
			s_map = s_map.to('cpu')
			kl += torch.sum(gt * torch.log(eps + gt/(s_map + eps)))
		return kl/bs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def bb_mapping(x_center_rel, y_center_rel, width_rel, height_rel, img_width = 64, img_height = 34):
    """
    Compute absolute bounding boxes values for given image size and given relative parameters

    :param x_center_rel: relative x value of bb center
    :param y_center_rel: relative y value of bb center
    :param width_rel: relative width
    :param height_rel: relative height
    :return: absolute values of bb borders
    """
    width_abs = width_rel*img_width
    height_abs = height_rel*img_height
    x_center_abs = x_center_rel*img_width
    y_center_abs = y_center_rel*img_height
    x_min = int(math.floor(x_center_abs - 0.5 * width_abs))
    x_max = int(math.floor(x_center_abs + 0.5 * width_abs))
    y_min = int(math.floor(y_center_abs - 0.5 * height_abs))
    y_max = int(math.floor(y_center_abs + 0.5 * height_abs))
    bb = [x if x>=0 else 0 for x in [x_min, x_max, y_min, y_max]]
    return bb

def visualization(heatmap,  gt, path,nr):
    heatmap = torchvision.transforms.functional.to_pil_image(heatmap)
    # gt = torchvision.transforms.functional.to_pil_image(gt)

    heatmap.save(os.path.join(path, '%s_pred.png'%nr))
    # gt.save(os.path.join(path, '%s_gt.png'%nr))