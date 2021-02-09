import albumentations as albu
import numpy as np
import segmentation_models_pytorch as smp

def get_training_augmentation():
    train_transform = [

#         albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0.1, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
#         albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_bounding_box(mask):
    maskx = np.any(mask, axis=0)
    masky = np.any(mask, axis=1)
    x1 = np.argmax(maskx)
    y1 = np.argmax(masky)
    x2 = len(maskx) - np.argmax(maskx[::-1])
    y2 = len(masky) - np.argmax(masky[::-1])
    # sub_image = image[y1:y2, x1:x2]
    return x1, y1, x2, y2


def get_model_class(model_name):
    DICT = {
        "unet": smp.Unet, 
        "unetplusplus": smp.UnetPlusPlus, 
        "manet": smp.MAnet, 
        "linknet": smp.Linknet, 
        "fpn": smp.FPN, 
        "pspnet": smp.PSPNet, 
        "pan": smp.PAN, 
        "deeplabv3": smp.DeepLabV3, 
        "deeplabv3plus": smp.DeepLabV3Plus
    }

    return DICT[model_name.lower()]