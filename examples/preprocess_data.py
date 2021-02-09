import json
import os

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_FOLDER = './data/images'
LABEL_FOLDER = './data/labels'

MASK_FOLDER = './data/masks'

if not os.path.exists(MASK_FOLDER):
    os.makedirs(MASK_FOLDER, exist_ok=True)

SEG_CLASSES = ['decision_box', 'retention_box', 'stamp_box']
for label_name in tqdm(os.listdir(LABEL_FOLDER)):
    label_path = os.path.join(LABEL_FOLDER, label_name)
    with open(label_path, 'r') as f:
        data = json.load(f)
        
    image_path = os.path.join(IMAGE_FOLDER, data['file_name'])
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
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
            
            if note == 'decision_box':
                mask[y: y + h, x: x + w] += 1
            elif note == 'retention_box':
                mask[y: y + h, x: x + w] += 2
            else:
                mask[y: y + h, x: x + w] += 3
    
    # import pdb; pdb.set_trace()
    cv2.imwrite(os.path.join(MASK_FOLDER, data['file_name']), mask)
