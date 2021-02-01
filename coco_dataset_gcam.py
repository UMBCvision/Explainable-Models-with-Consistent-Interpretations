import os
import random
import numpy as np
import pickle
from PIL import Image
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from pycocotools.coco import COCO


class CocoDetection(datasets.coco.CocoDetection):
    '''
        We introduce a custom dataloader for COCO dataset that returns both the original image and composite image.
        For each original image, we randomly pick one ground truth category from all the categories present in the image.
        We then select three negative images that do not contain the selected ground truth category and append them to the
        original image to create a composite image. We always compute Grad-CAM w.r.t to the selected ground truth category.
    '''
    def __init__(self, root, annFile, transform=None, target_transform=None, eval=False):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        if eval:
            self.negative_category_dict_img_list = pickle.load(open('coco_val_negative_category_dict_img_path_list.p', 'rb'))
        else:
            self.negative_category_dict_img_list = pickle.load(open('negative_category_dict_img_path_list.p', 'rb'))

        # we use x_start, x_end, y_start, y_end. For now, we assume each quadrant image will be 224x224
        self.quadrant_start_end_dict = {
            0: (0, 224, 0, 224),
            1: (0, 224, 224, 448),
            2: (224, 448, 0, 224),
            3: (224, 448, 224, 448)
        }

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        orig_height, orig_width = img.shape[-2], img.shape[-1]

        # get the annotations corresponding to this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        gt_indices = []
        output = torch.zeros((80), dtype=torch.long)
        for obj in target:
            gt_indices.append(self.cat2cat[obj['category_id']])
            output[gt_indices[-1]] = 1
        target = output

        # some images might not have annotations. we skip to the next index
        if len(gt_indices) == 0:
            return self.__getitem__(index+1)

        # randomly pick one GT category
        gcam_gt_category = random.choice(gt_indices)

        rand_im_list = []
        # pick 3 random images belonging to negative categories for the GT category
        rand_cat_im_path_list = self.negative_category_dict_img_list[gcam_gt_category]
        rand_cat_im_path_sublist = random.sample(rand_cat_im_path_list, 3)
        for rand_cat_im_path in rand_cat_im_path_sublist:
            rand_image = Image.open(os.path.join(self.root, rand_cat_im_path)).convert('RGB')
            rand_image = self.transform(rand_image)
            rand_im_list.append(rand_image)

        # select a random quadrant to place the GT image
        gt_quadrant = random.randint(0, 3)
        composite_image = torch.zeros((3, orig_height * 2, orig_width * 2), dtype=img.dtype)

        for quad_index in range(4):
            x_start, x_end, y_start, y_end = self.quadrant_start_end_dict[quad_index]
            if quad_index != gt_quadrant:
                im = rand_im_list.pop()
            else:
                im = img

            composite_image[:, x_start: x_end, y_start: y_end] = im[:, :, :]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, composite_image, gcam_gt_category, gt_quadrant
