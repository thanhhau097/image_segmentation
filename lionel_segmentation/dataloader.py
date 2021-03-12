import os

import cv2
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder, classes, size, augmentation, preprocessing):
        self.image_paths, self.mask_paths = self.get_input_paths(image_folder, mask_folder)
        # convert str names to class values on masks
        self.class_values = [classes.index(c) for c in classes]
        self.size = size
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def get_input_paths(self, image_folder, mask_folder):
        image_dict = dict((os.path.splitext(path)[0], os.path.join(image_folder, path)) for path in os.listdir(image_folder))
        mask_dict = dict((os.path.splitext(path)[0], os.path.join(mask_folder, path)) for path in os.listdir(mask_folder))

        joint_names = set(image_dict.keys()).intersection(set(mask_dict.keys()))

        image_paths, mask_paths = [], []
        for name in joint_names:
            image_paths.append(image_dict[name])
            mask_paths.append(mask_dict[name])

        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.resize(image, self.size)

        mask = cv2.imread(self.mask_paths[index], 0)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.uint8)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # import pdb; pdb.set_trace()
        return image, torch.tensor(mask, dtype=torch.int64)


class ChargridDataset(Dataset):
    def __init__(self, image_folder, mask_folder, classes, size, augmentation, preprocessing, charset):
        self.image_paths, self.mask_paths = self.get_input_paths(image_folder, mask_folder)
        # convert str names to class values on masks
        self.class_values = [classes.index(c) for c in classes]
        self.charset_values = [charset.index(c) for c in charset]
        self.size = size
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def get_input_paths(self, image_folder, mask_folder):
        image_dict = dict((os.path.splitext(path)[0], os.path.join(image_folder, path)) for path in os.listdir(image_folder))
        mask_dict = dict((os.path.splitext(path)[0], os.path.join(mask_folder, path)) for path in os.listdir(mask_folder))

        joint_names = set(image_dict.keys()).intersection(set(mask_dict.keys()))

        image_paths, mask_paths = [], []
        for name in joint_names:
            image_paths.append(image_dict[name])
            mask_paths.append(mask_dict[name])

        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # TODO: reduce loading, augmentation, preprocessing time
        # import time
        # start_time = time.time()

        chargrid = cv2.imread(self.image_paths[index], 0)
        chargrid = cv2.resize(chargrid, self.size, interpolation=cv2.INTER_NEAREST)
        chargrid = [(chargrid == v) for v in self.charset_values]
        chargrid = np.stack(chargrid, axis=-1).astype(np.uint8)
        # print('read chargrid', time.time() - start_time)
        # start_time = time.time()

        mask = cv2.imread(self.mask_paths[index], 0)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.uint8)
        # print('read mask', time.time() - start_time)
        # start_time = time.time()

        image = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
        # if self.augmentation:
        #     chargrid = self.augmentation(image=image, mask=chargrid)['mask']
        #     mask = self.augmentation(image=image, mask=mask)['mask']
        # print('augment time', time.time() - start_time)
        # start_time = time.time()    
        
        # apply preprocessing
        if self.preprocessing:
            chargrid = self.preprocessing(image=image, mask=chargrid)['mask']
            mask = self.preprocessing(image=image, mask=mask)['mask']
        
        # print('preprocessing time', time.time() - start_time)

        return chargrid, torch.tensor(mask, dtype=torch.int64)
