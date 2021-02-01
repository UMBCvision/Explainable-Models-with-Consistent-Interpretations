'''
    This code was adapted from https://github.com/allenai/elastic
    We introduce a custom CocoDetection class for creating our composite images.
    Our model uses both BCE and GCAM L1 consistency loss.
'''

import argparse
import time
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from PIL import Image
from utils import save_checkpoint, AverageMeter
from logger import Logger
import coco_dataset_gcam
import resnet_multigpu_multiclass as models


model_names = ['resnet18', 'resnet50', 'resnext50', 'resnext101']


class VanillaCocoDetection(datasets.coco.CocoDetection):
    '''
        This class is used as the dataloader for the COCO validation set.
    '''
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros(80, dtype=torch.long)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


parser = argparse.ArgumentParser(description='PyTorch COCO Training with Grad-CAM consistency')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=36, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=96, type=int,
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('-g', '--num-gpus', default=4, type=int,
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=117, type=int,
                    metavar='N', help='print frequency (default: 117)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--lambda', default=1, type=float,
                    metavar='LAM', help='lambda hyperparameter for GCAM loss', dest='lambda_val')
parser.add_argument('--save_dir', default='checkpoint', type=str, metavar='SV_PATH',
                    help='path to save checkpoints (default: none)')
parser.add_argument('--log_dir', default='logs', type=str, metavar='LG_PATH',
                    help='path to write logs (default: logs)')


def main():
    global args
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print('config: wd', args.weight_decay, 'lr', args.lr, 'batch_size', args.batch_size, 'num_gpus', args.num_gpus)
    logger = Logger(log_dir=args.log_dir)
    args_dict = vars(args)
    logger.add_hyperparams(args_dict)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model for COCO (80 classes)
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=80)

    model = torch.nn.DataParallel(model).cuda()

    bce_criterion = nn.BCEWithLogitsLoss().cuda()
    l1_criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.SGD([{'params': iter(params), 'lr': args.lr},
                                 ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            resume = ('module.fc.bias' in checkpoint['state_dict'] and
                      checkpoint['state_dict']['module.fc.bias'].size() == model.module.fc.bias.size()) or \
                     ('module.classifier.bias' in checkpoint['state_dict'] and
                      checkpoint['state_dict']['module.classifier.bias'].size() == model.module.classifier.bias.size())
            if resume:
                # True resume: resume training on COCO
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else print(
                    'no optimizer found')
                args.start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else args.start_epoch
            else:
                # Fake resume: transfer from ImageNet
                for n, p in list(checkpoint['state_dict'].items()):
                    if 'classifier' in n or 'fc' in n:
                        print(n, 'deleted from state_dict')
                        del checkpoint['state_dict'][n]
                model.load_state_dict(checkpoint['state_dict'], strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'] if 'epoch' in checkpoint else 'unknown'))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = coco_dataset_gcam.CocoDetection(os.path.join(args.data, 'train2014'),
                                                    os.path.join(args.data, 'annotations/instances_train2014.json'),
                                                    transforms.Compose([
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        normalize,
                                                    ]))
    val_dataset = VanillaCocoDetection(os.path.join(args.data, 'val2014'),
                                       os.path.join(args.data, 'annotations/instances_val2014.json'),
                                       transforms.Compose([
                                           transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))

    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate_multi(val_loader, model, bce_criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        coco_adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_multi(train_loader, model, bce_criterion, l1_criterion, optimizer, epoch)

        # evaluate on validation set
        validate_multi(val_loader, model, bce_criterion)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, args)


def train_multi(train_loader, model, bce_criterion, l1_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    bce_losses = AverageMeter()
    gc_losses = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    # switch to train mode
    model.train()
    optimizer.zero_grad()
    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    for i, (images, targets, composite_images, gcam_gt_targets, gt_quadrants) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        composite_images = composite_images.cuda(non_blocking=True)
        gcam_gt_targets = gcam_gt_targets.cuda(non_blocking=True)
        gt_quadrants = gt_quadrants.cuda(non_blocking=True)

        # compute output, BCE loss and GCAM loss.
        # We delegate multi-gpu GCAM loss handling to DataParallel by computing it within the forward pass.
        output, targets, bce_loss, gcam_loss = model(images, composite_images, targets, gcam_gt_targets,
                                                     gt_quadrants, bce_criterion, l1_criterion)

        bce_loss = bce_loss.mean()
        gcam_loss = gcam_loss.mean()
        bce_losses.update(bce_loss.item(), images.size(0))
        gc_losses.update(gcam_loss.item(), images.size(0))

        loss = bce_loss * 80.0 + args.lambda_val * gcam_loss

        # measure accuracy and record loss
        pred = output.data.gt(0.0).long()

        tp += (pred + targets).eq(2).sum(dim=0)
        fp += (pred - targets).eq(1).sum(dim=0)
        fn += (pred - targets).eq(-1).sum(dim=0)
        tn += (pred + targets).eq(0).sum(dim=0)
        count += images.size(0)

        this_tp = (pred + targets).eq(2).sum()
        this_fp = (pred - targets).eq(1).sum()
        this_fn = (pred - targets).eq(-1).sum()
        this_tn = (pred + targets).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        losses.update(float(loss), images.size(0))
        prec.update(float(this_prec), images.size(0))
        rec.update(float(this_rec), images.size(0))
        # compute gradient and do SGD step
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'BCE Loss {bce_loss.val:.4f} ({bce_loss.avg:.4f})\t'
                  'GC Loss {gc_loss.val:.4f} ({gc_loss.avg:.4f})\t'
                  'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, bce_loss=bce_losses, gc_loss=gc_losses, loss=losses, prec=prec, rec=rec))
            print('P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                  .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))


def validate_multi(val_loader, model, bce_criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    for i, (images, targets) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # original_target = target

        # compute output
        with torch.no_grad():
            output = model(images, eval=True)
            loss = bce_criterion(output, targets.float())

        # measure accuracy and record loss
        pred = output.data.gt(0.0).long()

        tp += (pred + targets).eq(2).sum(dim=0)
        fp += (pred - targets).eq(1).sum(dim=0)
        fn += (pred - targets).eq(-1).sum(dim=0)
        tn += (pred + targets).eq(0).sum(dim=0)
        # three_pred = pred.unsqueeze(1).expand(-1, 3, -1)  # n, 3, 80
        # tp_size += (three_pred + original_target).eq(2).sum(dim=0)
        # fn_size += (three_pred - original_target).eq(-1).sum(dim=0)
        count += images.size(0)

        this_tp = (pred + targets).eq(2).sum()
        this_fp = (pred - targets).eq(1).sum()
        this_fn = (pred - targets).eq(-1).sum()
        this_tn = (pred + targets).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        losses.update(float(loss), images.size(0))
        prec.update(float(this_prec), images.size(0))
        rec.update(float(this_rec), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                prec=prec, rec=rec))
            print('P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                  .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    print('--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))
    return


def coco_adjust_learning_rate(optimizer, epoch):
    if isinstance(optimizer, torch.optim.Adam):
        return
    lr = args.lr
    if epoch >= 24:
        lr *= 0.1
    if epoch >= 30:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
