import os
import math
import torch
from torch.utils.data import DataLoader
from data_reader import CUEING_reader
from torchvision.utils import save_image
from functions import *
from tqdm import tqdm
from token_network import Model

def clean(subset, data):
    grid = f'grids/grid1616/{subset}_grid.txt'
    img_dir = f'BDDA/{subset}/camera_images/'
    gaze_map = f'BDDA/{subset}/gazemap_images'

    task = 'main'
    th = 0.1

    dataset = CUEING_reader(task, data, subset, grid, img_dir, th, gaze_map)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=2, pin_memory=True)
    smoothing = GaussianSmoothing(1, 5, 1).cuda()
    heightfactor = 576 // 16
    widthfactor = 1024 // 16

    for i, (input, label, gaze_gt, img_name) in enumerate(tqdm(loader)):
        model.eval()


        with torch.no_grad():

            input = input.cuda()
            gaze_gt = gaze_gt.cuda()
            label = label.cuda()
            input += label
            mask = model(input)



            output = torch.sigmoid(mask)

            heatmap = grid2heatmap(output, [heightfactor, widthfactor], [16, 16])
            heatmap = F.interpolate(heatmap, size=[36, 64], mode='bilinear', align_corners=False)
            heatmap = smoothing(heatmap)
            heatmap = F.pad(heatmap, (2, 2, 2, 2), mode='constant')
            heatmap = heatmap.view(heatmap.size(0), -1)
            heatmap = F.softmax(heatmap, dim=1)

            # normalize
            heatmap -= heatmap.min(1, keepdim=True)[0]
            heatmap /= heatmap.max(1, keepdim=True)[0]

            heatmap = heatmap.view(-1, 1, 36, 64)

            for j in range(len(img_name)):
                im_name = img_name[j]
                gt_img = gaze_gt[j]
                pre = heatmap[j]
                ori = gaze_gt[j]

                loc =torch.where(pre<0.1)
                gt_img[loc] = 0


                vis = pre.detach()
                gt = gt_img.detach()
                ori = ori.detach()


                vis = vis.view(36,64,1)
                gt= gt.view(36, 64, 1)
                ori = ori.view(36, 64, 1)
                print(gt.shape)
                cv2.imshow('gt', np.array(gt))
                cv2.imshow('output', np.array(vis))
                cv2.imshow('ori_gt', np.array(ori))
                cv2.waitKey(0)


                # gaze_img_name = im_name.split('_')[0]+'_pure_hm_'+im_name.split('_')[1]
                #
                # gaze_map_filtered = os.path.join(f'BDDA/{subset}/fgazemap_images',gaze_img_name+'.jpg')
                # save_image(filtered_gaze, gaze_map_filtered,'jpeg')







if __name__ == "__main__":

    model = Model(channels=256, patch_size_W=80, patch_size_H=45, img_size_H=720, img_size_W=1280)
    model = model.cuda()
    model.load_state_dict(torch.load('CUEING-with-denoise'))

    data = 'BDDA'
    subset='testing'
    clean(subset,data)
