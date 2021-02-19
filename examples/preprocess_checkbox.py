import json
import os

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_FOLDER = './data/checkmark/images'
LABEL_FOLDER = './data/checkmark/labels'

CHECKMARK_REGION_IMAGE_FOLDER = './data/checkmark/region_images'
MASK_FOLDER = './data/checkmark/masks'

if not os.path.exists(MASK_FOLDER):
    os.makedirs(MASK_FOLDER, exist_ok=True)

if not os.path.exists(CHECKMARK_REGION_IMAGE_FOLDER):
    os.makedirs(CHECKMARK_REGION_IMAGE_FOLDER, exist_ok=True)

FIELDS = ['decision_classification', 'retention_period']
SEG_CLASSES = ['decision_box', 'retention_box']

for label_name in tqdm(os.listdir(LABEL_FOLDER)):
    label_path = os.path.join(LABEL_FOLDER, label_name)
    with open(label_path, 'r') as f:
        data = json.load(f)
        
    image_path = os.path.join(IMAGE_FOLDER, data['file_name'])
    image = cv2.imread(image_path)

    # create general mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for region in data['attributes']['_via_img_metadata']['regions']:
        formal_key = region['region_attributes']['formal_key']
        key_type = region['region_attributes']['key_type']

        if formal_key in FIELDS and key_type == 'value':
            shape = region['shape_attributes']
            if 'x' in shape:
                x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
            else:
                all_points_x = shape['all_points_x']
                all_points_y = shape['all_points_y']

                x = min(all_points_x)
                y = min(all_points_y)
                w = max(all_points_x) - min(all_points_x)
                h = max(all_points_y) - min(all_points_y)
            
            sub_image = image[y: y + h, x: x + w]
            mask[y: y + h, x: x + w] = 1

    print(data['file_name'], np.max(mask))
    if np.max(mask) == 0:
        continue
    # then cut out
    for region in data['attributes']['_via_img_metadata']['regions']:
        note = region['region_attributes']['note']
        if note in SEG_CLASSES:
            shape = region['shape_attributes']
            if 'x' in shape:
                x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
            else:
                all_points_x = shape['all_points_x']
                all_points_y = shape['all_points_y']

                x = min(all_points_x)
                y = min(all_points_y)
                w = max(all_points_x) - min(all_points_x)
                h = max(all_points_y) - min(all_points_y)
            sub_image = image[y: y + h, x: x + w]
            sub_mask = mask[y: y + h, x: x + w]
            
            name = os.path.splitext(data['file_name'])[0]
            cv2.imwrite(os.path.join(CHECKMARK_REGION_IMAGE_FOLDER, name + '_' + note + '.png'), sub_image)
            cv2.imwrite(os.path.join(MASK_FOLDER, name + '_' + note + '.png'), sub_mask)
