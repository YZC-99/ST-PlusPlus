import torch

from dataset.transform import crop, hflip, normalize, resize, blur, cutout

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import scipy.io

def preprocess_mask(img):
    od_mask = np.zeros_like(img)
    oc_mask = np.zeros_like(img)
    od_oc_mask = np.zeros_like(img)

    od_mask[img == 128] = 1
    od_mask[img == 0] = 1

    oc_mask[img == 0] = 1

    od_oc_mask[img == 128] = 1
    od_oc_mask[img == 0] = 2
    return {'refuge_od':od_mask,
            'refuge_oc':oc_mask,
            'refuge_od_oc':od_oc_mask}



class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        elif mode == 'src_tgt_train':
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids =  self.unlabeled_ids

        else:
            if mode == 'val':
                id_path = 'dataset/splits/%s/val.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))
        mask_path = os.path.join(self.root, id.split(' ')[1])

        if self.mode == 'val' or self.mode == 'label':
            if mask_path.endswith('.mat'):
                mask = scipy.io.loadmat(mask_path)['maskFull']
                mask = Image.fromarray(mask)
            else:
                mask = Image.open(mask_path)
            img, mask = resize(img, mask, 512)
            img, mask = normalize(img, mask)
            if mask_path.endswith('.tif'):
                od_mask = np.zeros_like(mask)
                od_mask[mask == 255] = 1
                mask = od_mask
            else:
                if self.name in ['refuge_od', 'refuge_oc', 'refuge_od_oc']:
                    masks = preprocess_mask(mask)
                    mask = masks[self.name]
                else:
                    masks = preprocess_mask(mask)
                    mask = masks['refuge_od']
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            if mask_path.endswith('.mat'):
                mask = scipy.io.loadmat(mask_path)['maskFull']
                mask = Image.fromarray(mask)
            else:
                mask = Image.open(mask_path)
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        base_size = 400 if self.name == 'pascal' else 2048
        img, mask = resize(img, mask, self.size)
        # img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' or self.mode == 'src_tgt_train' and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)
            img, mask = normalize(img, mask)
            od_mask = np.zeros_like(mask)
            od_mask[mask == 1] = 1
            return img, od_mask


        img, mask = normalize(img, mask)

        if mask_path.endswith('.tif'):
            od_mask = np.zeros_like(mask)
            od_mask[mask == 255] = 1
            mask = od_mask
        else:
            if self.name in ['refuge_od','refuge_oc','refuge_od_oc']:
                masks = preprocess_mask(mask)
                mask = masks[self.name]
            # elif self.name == 'refuge_domain':
            else:
                masks = preprocess_mask(mask)
                mask = masks['refuge_od']

        return img, mask

    def __len__(self):
        return len(self.ids)