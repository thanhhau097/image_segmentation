import os
import json

import cv2
import torch
import albumentations as albu
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

from lionel_segmentation.utils import to_tensor
from lionel_segmentation.utils import get_preprocessing
from lionel_segmentation.utils import get_validation_augmentation
from lionel_segmentation.utils import get_bounding_box


class SegmentationModel():
    def __init__(self, weights_path):
        ENCODER_WEIGHTS = 'imagenet'

        weights = torch.load(weights_path)
        self.model = smp.FPN(
            encoder_name=weights['encoder_name'], 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=weights['classes'], 
            activation=weights['activation']
        )

        self.model.load_state_dict(weights['state_dict'])
        self.size = weights['size']
        preprocessing_fn = smp.encoders.get_preprocessing_fn(weights['encoder_name'], ENCODER_WEIGHTS)
        self.preprocessing_fn = get_preprocessing()
        self.augmentation_fn = get_validation_augmentation()

    def process(self, input_image):
        """

        params input_image: image path or numpy image read by using cv2
        return: segmentation mask
        """
        raw_image = self.handle_input_image(input_image)
        image = cv2.resize(raw_image, (512, 512))
        image = self.augmentation_fn(image=image)['image']
        image = self.preprocessing_fn(image=image)['image']

        pr_mask = self.model.predict(image)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask = np.argmax(pr_mask, axis=0)

        pr_mask = cv2.resize(pr_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return pr_mask

    def process_and_visualize_bbox(self, input_image):
        raw_image = self.handle_input_image(input_image)

        mask = self.process(input_image)
        bb1 = get_bounding_box(mask == 1)
        bb2 = get_bounding_box(mask == 2)

        draw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(draw_image, (bb1[0], bb1[1]), (bb1[2], bb1[3]), (204, 102, 0), 2)
        cv2.rectangle(draw_image, (bb2[0], bb2[1]), (bb2[2], bb2[3]), (0, 255, 255), 2)

        return draw_image

    def handle_input_image(self, input_image):
        if type(input_image) == str:
            raw_image = cv2.imread(input_image)
        elif type(input_image) == np.ndarray:
            raw_image = input_image
        else:
            raise ValueError("only support image path or np.ndarray image")
        
        return raw_image


if __name__ == '__main__':
    model = SegmentationModel(weights_path='best_model.pth')