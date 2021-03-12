import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import pickle


charset_path = './data/project_charset.txt'
UNKNOWN_CHARACTER = '<UNK>'
CHARSET = [UNKNOWN_CHARACTER]
with open(charset_path, 'r') as f:
    chars = f.readlines()[0].replace('\n', '')

CHARSET += list(chars)

char2idx = dict((c, i) for i, c in enumerate(CHARSET))
idx2char = dict((i, c) for i, c in enumerate(CHARSET))

EXTRACTED_FIELDS = ['issue_name', 'date_status']
CLASSES = ['background', 'key_issue_name', 'key_date_status', 'value_issue_name', 'value_date_status']
cls2idx = dict((c, i) for i, c in enumerate(CLASSES))
idx2cls = dict((i, c) for i, c in enumerate(CLASSES))

IMAGE_FOLDER = './data/chargrid/images/'
LABEL_FOLDER = './data/chargrid/labels/'

CHARGRID_FOLDER = './data/chargrid/chargrid_images/'
MASK_FOLDER = './data/chargrid/masks/'

if not os.path.exists(CHARGRID_FOLDER):
    os.makedirs(CHARGRID_FOLDER, exist_ok=True)

if not os.path.exists(MASK_FOLDER):
    os.makedirs(MASK_FOLDER, exist_ok=True)


project_charset = dict()
for label_name in tqdm(os.listdir(LABEL_FOLDER)):
    label_path = os.path.join(LABEL_FOLDER, label_name)
    with open(label_path, 'r') as f:
        data = json.load(f)
        
    image_path = os.path.join(IMAGE_FOLDER, data['file_name'])
    image = cv2.imread(image_path)

    chargrid = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for region in data['attributes']['_via_img_metadata']['regions']:
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

        # make chargrid
        text = str(region['region_attributes']['label'])
        for c in text:
            if c not in project_charset:
                project_charset[c] = 1
            else:
                project_charset[c] += 1

        for i, char in enumerate(text):
            chargrid[y: y + h, x + i * (w // len(text)): x + (i + 1) * (w // len(text))] = char2idx.get(char, 0)

        # make mask
        formal_key = region['region_attributes']['formal_key']
        if formal_key in EXTRACTED_FIELDS:
            key_type = region['region_attributes']['key_type']
            mask[y: y + h, x: x + w] = cls2idx[key_type + '_' + formal_key]
    
    # import pdb; pdb.set_trace()
    # size = (512, 512)
    # chargrid = cv2.resize(chargrid, size, interpolation=cv2.INTER_NEAREST)
    # chargrid = [(chargrid == v) for v in range(len(CHARSET))]
    # chargrid = np.stack(chargrid, axis=-1).astype(np.uint8)
    # with open(os.path.join(CHARGRID_FOLDER, os.path.splitext(data['file_name'])[0] + '.pkl'), 'wb') as f:
    #     pickle.dump(chargrid, f)
    #     # np.save(f, chargrid)

    # mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    # masks = [(mask == v) for v in range(len(CLASSES))]
    # mask = np.stack(masks, axis=-1).astype(np.uint8)
    # with open(os.path.join(MASK_FOLDER, os.path.splitext(data['file_name'])[0] + '.pkl'), 'wb') as f:
    #     pickle.dump(mask, f)
    #     # np.save(f, mask)
    cv2.imwrite(os.path.join(CHARGRID_FOLDER, data['file_name']), chargrid)
    cv2.imwrite(os.path.join(MASK_FOLDER, data['file_name']), mask)


# with open('./data/project_charset.txt', 'w') as f:
#     f.write(''.join(list(dict(sorted(project_charset.items(), key=lambda item: -item[1])).keys())[:500]))