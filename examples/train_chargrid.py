import argparse

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from lionel_segmentation.dataloader import ChargridDataset
from lionel_segmentation.utils import get_training_augmentation
from lionel_segmentation.utils import get_validation_augmentation
from lionel_segmentation.utils import get_preprocessing
from lionel_segmentation.losses import GeneralizedDiceLossWrapper
from lionel_segmentation.utils import get_model_class


parser = argparse.ArgumentParser()
parser.add_argument("--train_image_folder", type=str, help="path to training image_folder")
parser.add_argument("--train_mask_folder", type=str, help="path to training mask_folder")
parser.add_argument("--val_image_folder", type=str, help="path to validation image_folder")
parser.add_argument("--val_mask_folder", type=str, help="path to validation mask_folder")
parser.add_argument("--classes", type=str, help="list of class to train, separated by comma")
parser.add_argument("--model", type=str, default='fpn', help="type of training model")
parser.add_argument("--encoder_name", type=str, default='se_resnext50_32x4d', help="type of encoder")
parser.add_argument("--encoder_weights", type=str, default='imagenet', help="type of encoder weights")
parser.add_argument("--size", type=int, default=512, help="training and evaluation size to scale")
parser.add_argument("--batch_size", type=int, default=8, help="training and evaluation batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--workers", type=int, default=8, help="number of training workers/cpus")
parser.add_argument("--pretrained_weights", type=str, help="path to pretrained weights")
parser.add_argument("--activation", type=str, default="softmax2d", help="type of activation function for the model, softmax2d/None/sigmoid")
parser.add_argument("--loss", type=str, default="gdl", help="loss function to use")
parser.add_argument("--chargrid", action='store_true', help='whether or not train chargrid model')
parser.add_argument("--charset_path", type=str, help="path to charset file")

args = parser.parse_args()


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CREATE MODEL
    if args.chargrid:
        CLASSES = ["unlabelled"] + ['key_' + c for c in args.classes.split(',')] + ['value_' + c for c in args.classes.split(',')]
        charset = ['<UNK>']
        with open(args.charset_path, 'r') as f:
            chars = f.readlines()[0].replace('\n', '')
        charset += list(chars)
    else:
        CLASSES = ["unlabelled"] + args.classes.split(',')
        charset = 'rgb'

    print('training classes:', CLASSES)

    model_class = get_model_class(args.model)
    model = model_class(
        in_channels=len(charset),
        encoder_name=args.encoder_name, 
        encoder_weights=args.encoder_weights, 
        classes=len(CLASSES),
        activation=args.activation,
    )
    
    # load pretrained weights
    if args.pretrained_weights:
        weights = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(weights['state_dict'])

    model = torch.nn.DataParallel(model)

    # DATA
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name, args.encoder_weights)

    train_dataset = ChargridDataset(
        args.train_image_folder, 
        args.train_mask_folder, 
        size=(args.size, args.size),
        classes=CLASSES,
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        charset=charset
    )

    valid_dataset = ChargridDataset(
        args.val_image_folder, 
        args.val_mask_folder,
        classes=CLASSES,
        size=(args.size, args.size),
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        charset=charset
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
        dict(params=model.parameters(), lr=args.lr),
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

    for i in range(0, args.epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']

            save_dict = {
                'state_dict': model.module.state_dict(),
                'model': args.model,
                'encoder_name': args.encoder_name,
                'encoder_weights': args.encoder_weights,
                'classes': CLASSES,
                'charset': charset,
                'activation': args.activation, 
                'size': args.size
            }
            torch.save(save_dict, './weights/chargrid_best_model.pth')
            print('Model saved!')
        

train()
