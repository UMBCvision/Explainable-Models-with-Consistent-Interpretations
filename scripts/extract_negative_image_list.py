import os
import pickle
from pycocotools.coco import COCO
from collections import defaultdict


def extract_coco_negative_image_list(coco, ids, cat_mapping_dict):
    """
        This function creates a dictionary such that each ground truth category index has an associated
        list of negative images which do not contain the ground truth category
    :param coco: coco dataset instance from pycocotools
    :param ids: image keys corresponding to each image in the coco dataset instance
    :param cat_mapping_dict: dictionary mapping the COCO ground truth indices from 90 to 80 actual categories
    :return: A dictionary containing a negative list of images for each ground truth category
    """

    COCO_NUM_CATEGORIES = 80
    negative_category_dict_img_list = defaultdict(list)

    # We will iterate over every annotation for a given image and add the image to the negative set of absent categories
    for idx, image_id in enumerate(ids):
        im_path = coco.loadImgs(image_id)[0]['file_name']
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotation_list = coco.loadAnns(ann_ids)
        curr_cat_list = [cat_mapping_dict[ann_obj['category_id']] for ann_obj in annotation_list]
        # Skip the current image if there are no GT annotations
        if len(curr_cat_list) == 0:
            continue
        curr_cat_set = set(curr_cat_list)
        gt_cat_set = set(range(COCO_NUM_CATEGORIES))
        negative_gt_set = gt_cat_set.difference(curr_cat_set)

        for negative_gt in negative_gt_set:
            negative_category_dict_img_list[negative_gt].append(im_path)

        if (idx + 1) % 100 == 0:
            print('Finished processing {} images'.format(idx + 1))
    return negative_category_dict_img_list


# Replace the below path with root path for the COCO dataset
coco_root = '/home/vipin/coco/'
coco_train = COCO(os.path.join(coco_root, 'annotations/instances_train2014.json'))
coco_val = COCO(os.path.join(coco_root, 'annotations/instances_val2014.json'))
COCO_NUM_CATEGORIES = 80

train_ids = list(coco_train.imgs.keys())
val_ids = list(coco_val.imgs.keys())

# map the categories from 1-90 (which have missing intermediate category indices) to range 0-79
cat_mapping_90_to_80 = dict()
for cat in coco_train.cats.keys():
    cat_mapping_90_to_80[cat] = len(cat_mapping_90_to_80)

# Extract and write the negative dictionary list for COCO training images
negative_category_dict_img_list = extract_coco_negative_image_list(coco_train, train_ids, cat_mapping_90_to_80)
pickle.dump(negative_category_dict_img_list, open('negative_category_dict_img_path_list.p', 'wb'))

# Extract and write the negative dictionary list for COCO validation images
negative_category_dict_val_img_list = extract_coco_negative_image_list(coco_val, val_ids, cat_mapping_90_to_80)
pickle.dump(negative_category_dict_val_img_list, open('coco_val_negative_category_dict_img_path_list.p', 'wb'))
