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

from data_reader import CUEING_reader
from model import Model
import cv2
from sklearn.metrics import f1_score,precision_score,recall_score, roc_curve, roc_auc_score
from functions import *



parser = argparse.ArgumentParser(description='Feature Training and Test')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--no_train', action='store_true', default=False)
parser.add_argument('--weight', metavar='DIR', help='path to load the weight')

parser.add_argument('--finetune', action='store_true', default=False)
parser.add_argument('--fineweight', metavar='DIR', help='path to load the fintune weight')


parser.add_argument('--gridheight', default=16, type=int, metavar='N',
                    help='number of rows in grid')
parser.add_argument('--gridwidth', default=16, type=int, metavar='N',
                    help='number of columns in grid ')

parser.add_argument('--traindir', metavar='DIR', help='path to images folder')
parser.add_argument('--testdir', metavar='DIR', help='path to images folder')
parser.add_argument('--validir', metavar='DIR', help='path to images folder')

parser.add_argument('--traingazemaps', metavar='DIR', help='path to gaze map images folder')
parser.add_argument('--valigazemaps', metavar='DIR', help='path to gaze map images folder')
parser.add_argument('--testgazemaps', metavar='DIR', help='path to gaze map images folder')

parser.add_argument('--traingrid', default='', type=str, metavar='PATH', help='path to txt with grid entries for training images')
parser.add_argument('--valgrid', default='', type=str, metavar='PATH', help='path to txt with grid entries for validation images')
parser.add_argument('--testgrid', default='', type=str, metavar='PATH', help='path to txt with grid entries for test images')


parser.add_argument('--yolo5bbtest', metavar='DIR', help='path to folder of yolo5 bounding box txt files')

parser.add_argument('--visualizations', action='store_true', default=False)
parser.add_argument('--threshhold', default=0.5, type=float, metavar='N', help='threshold for object-level evaluation')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    shutup.please()
    args = parser.parse_args()

    train_writer = SummaryWriter(log_dir='loss/train')
    vali_writer = SummaryWriter(log_dir='loss/validation')


    dim = args.gridwidth*args.gridheight
    th = 1/dim

    torch.manual_seed(20)
    torch.cuda.manual_seed(20)
    np.random.seed(20)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = Model(channels=16, patch_size_W=80, patch_size_H=45, img_size_H=720, img_size_W=1280)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    task = 'main'

    if not args.no_train:


        train_dataset = CUEING_reader(task, "training", args.traingrid, args.traindir, th, args.traingazemaps)
        # val_dataset = CUEING_reader(task, "validation", args.valgrid, args.validir, th, args.valigazemaps)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle= True,
            num_workers=2, pin_memory=True)

        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset,
        #     batch_size=args.batch_size, shuffle=True,
        #     num_workers=2, pin_memory=True)

    best_loss= 10000
    if not args.no_train:
        if args.finetune:
            model.load_state_dict(torch.load(args.fineweight))
        for epoch in range(args.epochs):

            adjust_learning_rate(optimizer, epoch, args)

            train(train_loader, criterion, optimizer,  epoch, args, model,train_writer)
            # vali_loss = validate(val_loader,args, model, criterion,  epoch, vali_writer)

            # if best_loss > vali_loss:
            #     best_loss = vali_loss
            torch.save(model.state_dict(), 'weight')
            print('Weight Saved!')
        train_writer.close()
        vali_writer.close()


    else:

        test_dataset = CUEING_reader(task, "testing", args.testgrid, args.testdir, th, args.testgazemaps)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True)
        model.load_state_dict(torch.load(args.weight))
        test(test_loader, model, args)

def train(train_loader,  criterion, optimizer, epoch, args, main_model,train_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    main_model.train()
    end = time.time()

    for i, (input, grid, gaze_gt, img_name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            grid = grid.cuda(args.gpu, non_blocking=True)

        mask = main_model(input)
        loss_attention2 = criterion(mask, grid)


        total_loss = loss_attention2
        # tensorboard
        train_writer.add_scalar(tag="loss/train", scalar_value=total_loss,
                          global_step=epoch * len(train_loader) + i)


        losses.update(total_loss.item(), len(img_name))
        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

def validate(val_loader, args, main_model, criterion,  epoch,vali_writer):

    main_model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    kld_losses = AverageMeter()
    cc_losses = AverageMeter()
    end = time.time()

    heightfactor = 576 // args.gridheight
    widthfactor = 1024 // args.gridwidth

    smoothing = GaussianSmoothing(1, 5, 1).cuda(args.gpu)
    with torch.no_grad():
        for i, (input, grid, gaze_gt, img_name) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                gaze_gt = gaze_gt.cuda(args.gpu, non_blocking=True)
                grid = grid.cuda(args.gpu, non_blocking=True)



            mask = main_model(input)


            loss_attention2 = criterion(mask, grid)
            total_loss = loss_attention2

            vali_writer.add_scalar(tag="loss/validation", scalar_value=total_loss,
                          global_step=epoch * len(val_loader) + i)


            losses.update(total_loss.item(), len(img_name))

            batch_time.update(time.time() - end)
            end = time.time()



            if i % args.print_freq == 0:
                output = torch.sigmoid(mask)

                heatmap = grid2heatmap(output, [heightfactor, widthfactor], [args.gridheight, args.gridwidth])
                heatmap = F.interpolate(heatmap, size=[36, 64], mode='bilinear', align_corners=False)
                heatmap = smoothing(heatmap)
                heatmap = F.pad(heatmap, (2, 2, 2, 2), mode='constant')
                heatmap = heatmap.view(heatmap.size(0), -1)
                heatmap = F.softmax(heatmap, dim=1)

                # normalize
                heatmap -= heatmap.min(1, keepdim=True)[0]
                heatmap /= heatmap.max(1, keepdim=True)[0]

                heatmap = heatmap.view(-1, 1, 36, 64)



                kld = kl(heatmap, gaze_gt)

                c = cc(heatmap, gaze_gt)

                if math.isnan(kld)==False and math.isnan(c)==False:
                    kld_losses.update(kld, len(img_name))
                    cc_losses.update(c, len(img_name))

                print('validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'KL {kl.val:.4f} ({kl.avg:.4f})\t'
                      'CC {cc.val:.4f} ({cc.avg:.4f})\t'
                    .format(
                    i, len(val_loader), loss =losses ,kl=kld_losses, cc=cc_losses))
    return total_loss

def test(test_loader, main_model, args):
    main_model.eval()

    batch_time = AverageMeter()
    kld_losses = AverageMeter()
    cc_losses = AverageMeter()

    end = time.time()

    heightfactor = 576 // args.gridheight
    widthfactor = 1024 // args.gridwidth

    smoothing = GaussianSmoothing(1, 5, 1).cuda(args.gpu)


    tp = 0
    fp = 0
    fn = 0
    all_count = 0
    hm_max_values = []
    gt = []

    with torch.no_grad():
        for i, (input,  gaze_gt, img_name) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                gaze_gt = gaze_gt.cuda(args.gpu, non_blocking=True)
                # label = label.cuda(args.gpu, non_blocking=True)



            # input +=label
            mask = main_model(input)


            batch_time.update(time.time() - end)
            end = time.time()




            output = torch.sigmoid(mask)

            heatmap = grid2heatmap(output, [heightfactor, widthfactor], [args.gridheight, args.gridwidth])
            heatmap = F.interpolate(heatmap, size=[36, 64], mode='bilinear', align_corners=False)
            heatmap = smoothing(heatmap)
            heatmap = F.pad(heatmap, (2, 2, 2, 2), mode='constant')
            heatmap = heatmap.view(heatmap.size(0), -1)
            heatmap = F.softmax(heatmap, dim=1)

            # normalize
            heatmap -= heatmap.min(1, keepdim=True)[0]
            heatmap /= heatmap.max(1, keepdim=True)[0]

            heatmap = heatmap.view(-1, 1, 36, 64)


            kld = kl(heatmap, gaze_gt)
            c = cc(heatmap, gaze_gt)
            if math.isnan(kld) == False and math.isnan(c) == False:
                kld_losses.update(kld, len(img_name))
                cc_losses.update(c, len(img_name))
            if i % args.print_freq == 0:
                print('test: [{0}/{1}]\t'
                      'KL {kl.val:.4f} ({kl.avg:.4f})\t'
                      'CC {cc.val:.4f} ({cc.avg:.4f})\t'
                    .format(
                    i, len(test_loader), kl=kld_losses, cc=cc_losses))
            for j in range(len(img_name)):
                img_names = img_name[j]
                heatmap_img = heatmap[j]  # predicted gaze map
                gt_img = gaze_gt[j]  # original gaze map
                if args.visualizations:
                    visualization(heatmap_img.cpu(), gt_img.cpu(), 'visualizations', img_names)


                filename = os.path.join(args.yolo5bbtest, img_names + ".txt")

                if os.path.exists(filename):
                    with open(filename) as f:
                        for linestring in f:
                            all_count += 1

                            line = linestring.split()

                            width = float(line[3])
                            height = float(line[4])
                            x_center = float(line[1])
                            y_center = float(line[2])

                            x_min, x_max, y_min, y_max = bb_mapping(x_center, y_center, width, height, img_width = 64, img_height = 36)

                            # find maximum pixel value within object bounding box
                            gt_obj = gt_img[0, y_min:y_max + 1, x_min:x_max + 1]

                            gt_obj_max = torch.max(gt_obj)
                            heatmap_obj = heatmap_img[0, y_min:y_max + 1, x_min:x_max + 1]
                            heatmap_obj_max = torch.max(heatmap_obj)

                            # object is recognized if maximum pixel value is higher than th

                            gt_obj_recogn = gt_obj_max > 0.15


                            hm_obj_recogn = heatmap_obj_max >0.5
                            hm_max_values.append(heatmap_obj_max)

                            if gt_obj_recogn:
                                gt.append(1)
                            else:
                                gt.append(0)

                            if (hm_obj_recogn and gt_obj_recogn):
                                tp += 1
                            elif (hm_obj_recogn and not gt_obj_recogn):
                                fp += 1
                            elif (not hm_obj_recogn and gt_obj_recogn):
                                fn += 1




        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        tn = all_count - tp - fp - fn
        acc = (tp + tn) / all_count

        f1 = 2 * precision * recall / (precision + recall)
        print('Object-level results:')
        print('tp:', tp, 'fp:', fp, 'tn:', tn, 'fn:', fn, 'sum:', all_count)
        print('prec:', precision, 'recall:', recall, 'f1', f1, 'acc', acc)
        print('AUC:', roc_auc_score(gt, hm_max_values))





if __name__ == '__main__':
    main()