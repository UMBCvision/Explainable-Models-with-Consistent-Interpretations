import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import resnet_multigpu as resnet
import resnet_multigpu_maxpool as resnet_max
import os
import cv2
import datasets as pointing_datasets
from pointing_game import PointingGameBenchmark

""" 
    Here, we evaluate Stochastic Pointing Game (SPG) on MS-COCO dataset 
    by using the segmentation mask annotations on the val dataset.
"""

model_names = ['resnet18', 'resnet50', 'alexnet']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-g', '--num-gpus', default=1, type=int,
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input_resize', default=224, type=int,
                    metavar='N', help='Resize for smallest side of input (default: 224)')
parser.add_argument('--maxpool', dest='maxpool', action='store_true',
                    help='use maxpool version of the model')

def main():
    global args
    args = parser.parse_args()

    if args.maxpool:
        print("=> creating maxpool version of model '{}'".format(args.arch))
        model = resnet_max.__dict__[args.arch](num_classes=80)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch](num_classes=80)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if (not args.resume) and (not args.pretrained):
        assert False, "Please specify either the pre-trained model or checkpoint for evaluation"

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # Here, we don't resize the images. Instead, we feed the full image and use AdaptivePooling before FC.
    # We will resize Gradcam heatmap to image size and compare the actual mask
    val_dataset = pointing_datasets.CocoDetection(os.path.join(args.data, 'val2014'),
                                                  os.path.join(args.data, 'annotations/instances_val2014.json'),
                                                  transform=transforms.Compose([
                                                      transforms.Resize(args.input_resize),
                                                      transforms.ToTensor(),
                                                      normalize,
                                                  ]))

    # we set batch size=1 since we are loading full resolution images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=val_dataset.collate)

    validate_multi(val_loader, val_dataset, model)


def validate_multi(val_loader, val_dataset, model):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    expected_value_list = []
    NUM_SAMPLES = 100
    TOLERANCE = 15
    pointinggame = PointingGameBenchmark(val_dataset)
    end = time.time()
    for i, (images, annotation) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)

        # compute output
        # we have added return_feats=True to get the output as well as layer4 conv feats
        output, feats = model(images, return_feats=True)
        class_ids = val_dataset.as_class_ids(annotation[0])
        if (len(annotation[0]) == 0) or annotation[0] is None:
            # this image lacks annotation and hence we skip this
            continue
        else:
            w, h = val_dataset.as_image_size(annotation[0])

        # Now, we iterate over every GT category
        for class_id in class_ids:
            output_gradcam = compute_gradcam(output, feats, class_id)
            output_gradcam_np = output_gradcam.data.cpu().numpy()[0]    # since we have batch size==1
            resized_output_gradcam = cv2.resize(output_gradcam_np, (w, h))
            spatial_sum = resized_output_gradcam.sum()
            if spatial_sum <= 0:
                # We ignore images with zero Grad-CAM
                continue
            resized_output_gradcam = resized_output_gradcam / spatial_sum

            gt_mask = pointing_datasets.coco_as_mask(val_dataset, annotation[0], class_id)

            # output_gradcam is now normalized and can be considered as probabilities
            # We sample a point on the GCAM mask using the normalized GCAM as probabilities
            # sample_index = np.random.choice(np.arange(h*w), p=resized_output_gradcam.ravel())
            sample_indices = np.random.choice(np.arange(h * w), NUM_SAMPLES, p=resized_output_gradcam.ravel())
            curr_image_hits = []
            for sample_index in sample_indices:
                sample_x, sample_y = np.unravel_index(sample_index, (h, w))
                v, u = torch.meshgrid((
                    (torch.arange(gt_mask.shape[0],
                                  dtype=torch.float32) - sample_x) ** 2,
                    (torch.arange(gt_mask.shape[1],
                                  dtype=torch.float32) - sample_y) ** 2,
                ))
                accept = (v + u) < TOLERANCE ** 2
                hit = (gt_mask & accept).view(-1).any()
                if hit:
                    hit = +1
                else:
                    hit = -1
                # hit = pointinggame.evaluate(annotation[0], class_id, (sample_x, sample_y))
                curr_image_hits.append((hit + 1) / 2)
            curr_image_hits_arr = np.array(curr_image_hits)
            # we have a bernoulli distribution for the hits, so we compute mean and variance
            pos_prob = float(curr_image_hits_arr.sum()) / float(NUM_SAMPLES)
            # expected_value_list.append(pos_prob)
            expected_value_list.append(pos_prob*100)    # We need % for mean

        if i % 1000 == 0:
            print('\nCOCO stochastic pointing game results after {} examples: '.format(i+1))
            expected_value_arr = np.array(expected_value_list)
            mean_expectation = expected_value_arr.mean()
            print('Mean - Expected value for 100 stochastic samples/image for hits(1) and misses(0): {}'.format(
                mean_expectation))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\n\n')
    expected_value_arr = np.array(expected_value_list)
    mean_expectation = expected_value_arr.mean()
    print('Mean - Expected value for 100 stochastic samples/image for hits(1) and misses(0): {}'.format(
        mean_expectation))

    return


def compute_gradcam(output, feats, target):
    """
    Compute the gradcam for the given target
    :param output:
    :param feats:
    :param: target:
    :return:
    """
    eps = 1e-8
    relu = nn.ReLU(inplace=True)

    target = target.cpu().numpy()
    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.requires_grad = True

    # Compute the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.cuda() * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).cuda(),
                                  retain_graph=True, create_graph=True)
    # Changing to dot product of grad and features to preserve grad spatial locations
    gcam512_1 = dy_dz1 * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = relu(gradcam)
    spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam = (gradcam / (spatial_sum1 + eps)) + eps

    return gradcam


def unwrap_dict(dict_object):
    new_dict = {}
    for k, v in dict_object.items():
        if k == 'object':
            new_v_list = []
            for elem in v:
                new_v_list.append(unwrap_dict(elem))
            new_dict[k] = new_v_list
            continue
        if isinstance(v, dict):
            new_v = unwrap_dict(v)
        elif isinstance(v, list) and len(v) == 1:
            new_v = v[0]
        else:
            new_v = v
        new_dict[k] = new_v
    return new_dict


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


if __name__ == '__main__':
    main()
