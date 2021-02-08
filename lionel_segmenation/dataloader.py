import os

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder, classes, size, augmentation, preprocessing):
        self.image_paths, self.mask_paths = self.get_input_paths(image_folder, mask_folder)
        # convert str names to class values on masks
        self.class_values = [classes.index(c.lower()) for c in classes]
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
        
        return image, torch.tensor(mask, dtype=torch.int64)