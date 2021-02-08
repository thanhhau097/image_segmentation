import argparse

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from lionel_segmenation.dataloader import SegmentationDataset
from lionel_segmenation.utils import get_training_augmentation
from lionel_segmenation.utils import get_validation_augmentation
from lionel_segmenation.utils import get_preprocessing
from lionel_segmenation.losses import GeneralizedDiceLossWrapper


parser = argparse.ArgumentParser()
parser.add_argument("--train_image_folder", type=str, help="path to training image_folder")
parser.add_argument("--train_mask_folder", type=str, help="path to training mask_folder")
parser.add_argument("--val_image_folder", type=str, help="path to validation image_folder")
parser.add_argument("--val_mask_folder", type=str, help="path to validation mask_folder")
parser.add_argument("--classes", type=str, help="list of class to train, separated by comma")
parser.add_argument("--encoder_name", type=str, default='se_resnext50_32x4d', help="type of encoder")
parser.add_argument("--size", type=int, default=512, help="training and evaluation size to scale")
parser.add_argument("--batch_size", type=int, default=8, help="training and evaluation batch size")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--workers", type=int, default=8, help="number of training workers/cpus")
parser.add_argument("--pretrained_weights", type=str, help="path to pretrained weights")
parser.add_argument("--activation", type=str, default="softmax2d", help="type of activation function for the model, softmax2d/None/sigmoid")
parser.add_argument("--loss", type=str, default="gdl", help="loss function to use")
args = parser.parse_args()


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CREATE MODEL
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = ["unlabelled"] + args.classes.split(',')

    model = smp.FPN(
        encoder_name=args.encoder_name, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=args.activation,
    )
    model = torch.nn.DataParallel(model)
    
    # load pretrained weights
    if args.pretrained_weights:
        weights = torch.load('./best_model.pth')
        model.load_state_dict(weights['state_dict'])

    # DATA
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, ENCODER_WEIGHTS)

    train_dataset = SegmentationDataset(
        args.train_image_folder, 
        args.train_mask_folder, 
        size=(args.size, args.size),
        classes=CLASSES,
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = SegmentationDataset(
        args.val_image_folder, 
        args.val_mask_folder,
        classes=CLASSES,
        size=(args.size, args.size),
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # LOSS AND METRICS
    if args.loss.lower() == 'gdl':
        loss = GeneralizedDiceLossWrapper()
    else:
        raise ValueError("Currently only support Generalized Dice Loss")

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # TRAINING:
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    max_score = 0

    for i in range(0, 5000):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']

            save_dict = {
                'state_dict': model.module.state_dict(),
                'encoder_name': args.encoder_name,
                'classes': CLASSES,
                'activation': args.activation, 
                'size': args.size
            }
            torch.save(save_dict, './best_model.pth')
            print('Model saved!')
        

train()
