from vision import VisionDataset

from PIL import Image

import os
import os.path
import sys
import random
import numpy as np
import torch
import pdb

""" We have modified the dataloader such that along with the original image and the corresponding target, we also
    return a composite image of 2x2 grid, containing 3 other images chosen from a random category excluding the target.
    Although we return the ground truth quadrant for the 
    positive image in the 2x2 composite image, we do not currently use it. Instead, we split the batch sequentially 
    corresponding to the 4 quadrants. This should not affect the loss since the images themselves are shuffled within 
    the batch and the loss is computed as the mean over the batch.
    
    This code is specifically for an image of resolution - 224x224 and composite image of resolution - 448x448.  
"""


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    target_idx_to_im_path_dict = {}
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        target_idx = class_to_idx[target]
        if target_idx not in target_idx_to_im_path_dict:
            target_idx_to_im_path_dict[target_idx] = []
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, target_idx)
                    images.append(item)

                    # Also add to a dictionary mapping the target to the image path - vipin
                    target_idx_to_im_path_dict[target_idx].append(path)

    return images, target_idx_to_im_path_dict


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples, target_idx_to_im_path_dict = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.category_indices = np.arange(1000)     # hard-coded for imagenet num classes
        self.target_idx_to_im_path_dict = target_idx_to_im_path_dict    # mapping from target idx to im paths

        # we use x_start, x_end, y_start, y_end
        self.quadrant_start_end_dict = {
            0: (0, 224, 0, 224),
            1: (0, 224, 224, 448),
            2: (224, 448, 0, 224),
            3: (224, 448, 224, 448)
        }
        self.neg_category_indices = list(range(1000))

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        orig_height, orig_width = sample.shape[-2], sample.shape[-1]

        # We need to return the original image, original target, composite image, GT quadrant for original
        # pick 3 other categories to create the composite image
        negative_category_indices = self.neg_category_indices.copy()

        # remove the GT category from neg list
        del negative_category_indices[negative_category_indices.index(target)]
        rand_cat_indices = np.random.choice(negative_category_indices, 3)
        rand_im_list = []
        for rand_cat_index in rand_cat_indices:
            rand_cat_im_path_list = self.target_idx_to_im_path_dict[rand_cat_index]
            rand_cat_im_path = random.choice(rand_cat_im_path_list)
            rand_image = self.loader(rand_cat_im_path)
            rand_image = self.transform(rand_image)
            rand_im_list.append(rand_image)

        gt_quadrant = random.randint(0, 3)
        composite_image = torch.zeros((3, orig_height*2, orig_width*2), dtype=sample.dtype)

        for quad_index in range(4):
            x_start, x_end, y_start, y_end = self.quadrant_start_end_dict[quad_index]
            if quad_index != gt_quadrant:
                im = rand_im_list.pop()
            else:
                im = sample

            composite_image[:, x_start: x_end, y_start: y_end] = im[:, :, :]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, composite_image, gt_quadrant

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
