import os
import json
import random

import cv2
import torch
import albumentations as albu
import numpy as np
import segmentation_models_pytorch as smp

from lionel_segmentation.utils import to_tensor
from lionel_segmentation.utils import get_preprocessing
from lionel_segmentation.utils import get_validation_augmentation
from lionel_segmentation.utils import get_bounding_box
from lionel_segmentation.utils import get_model_class


class SegmentationModel():
    def __init__(self, weights_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        weights = torch.load(weights_path, map_location=self.device)
        model_class = get_model_class(weights['model'])
        self.model = model_class(
            encoder_name=weights['encoder_name'], 
            encoder_weights=weights['encoder_weights'], 
            classes=len(weights['classes']), 
            activation=weights['activation']
        )
        self.model = self.model.to(self.device)

        self.classes = weights['classes']
        self.model.load_state_dict(weights['state_dict'])
        self.size = weights['size']
        preprocessing_fn = smp.encoders.get_preprocessing_fn(weights['encoder_name'], weights['encoder_weights'])
        self.preprocessing_fn = get_preprocessing(preprocessing_fn)
        self.augmentation_fn = get_validation_augmentation()

    def process(self, input_image):
        """

        params input_image: image path or numpy image read by using cv2
        return: segmentation mask
        """
        raw_image = self.handle_input_image(input_image)
        image = cv2.resize(raw_image, (self.size, self.size))
        image = self.augmentation_fn(image=image)['image']
        image = self.preprocessing_fn(image=image)['image']
        image = torch.from_numpy(image).to(self.device).unsqueeze(0)

        pr_mask = self.model.predict(image)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask = np.argmax(pr_mask, axis=0)

        pr_mask = cv2.resize(pr_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return pr_mask

    def process_and_visualize_bbox(self, input_image):
        raw_image = self.handle_input_image(input_image)

        mask = self.process(input_image)
        
        draw_image = raw_image.copy()
        for i in range(1, len(self.classes)):
            bb = get_bounding_box(mask == i)
            cv2.rectangle(draw_image, (bb[0], bb[1]), (bb[2], bb[3]), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 3)
            
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
    from tqdm import tqdm

    model = SegmentationModel(weights_path='weights/best_model.pth')
    IMAGE_FOLDER = '/mnt/ai_filestore/home/lionel/research/image_segmentation/data/val_images/'
    for image_name in tqdm(os.listdir(IMAGE_FOLDER)):
        draw_image = model.process_and_visualize_bbox(os.path.join(IMAGE_FOLDER, image_name))
        cv2.imwrite('data/debugs/{}'.format(image_name), draw_image)