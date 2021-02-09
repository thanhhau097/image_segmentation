# image_segmentation

# INTRODUCTION

# TRAINING
1. Change the list of augmentation functions in utils.py depends on your problems
2. Training:
```
CUDA_VISIBLE_DEVICES=0,3 python lionel_segmentation/train.py --train_image_folder=./data/images --train_mask_folder=./data/masks/ --val_image_folder=./data/images/ --val_mask_folder=./data/masks/ --classes=decision_box,retention_box --workers=8 --pretrained_weights=./best_model.pth
```

# LIST OF MODELS AND ENCODERS (continue updating on https://github.com/qubvel/segmentation_models.pytorch)

### MODELS:
Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus

### ENCODERS and WEIGHTS
🏔 Available Encoders
=====================

ResNet
~~~~~~~~~~

+-------------+-------------------------+-------------+
| Encoder     | Weights                 | Params, M   |
+=============+=========================+=============+
| resnet18    | imagenet / ssl / swsl   | 11M         |
+-------------+-------------------------+-------------+
| resnet34    | imagenet                | 21M         |
+-------------+-------------------------+-------------+
| resnet50    | imagenet / ssl / swsl   | 23M         |
+-------------+-------------------------+-------------+
| resnet101   | imagenet                | 42M         |
+-------------+-------------------------+-------------+
| resnet152   | imagenet                | 58M         |
+-------------+-------------------------+-------------+
~~~~~~~~~~
ResNeXt
~~~~~~~~~~

+----------------------+-------------------------------------+-------------+
| Encoder              | Weights                             | Params, M   |
+======================+=====================================+=============+
| resnext50\_32x4d     | imagenet / ssl / swsl               | 22M         |
+----------------------+-------------------------------------+-------------+
| resnext101\_32x4d    | ssl / swsl                          | 42M         |
+----------------------+-------------------------------------+-------------+
| resnext101\_32x8d    | imagenet / instagram / ssl / swsl   | 86M         |
+----------------------+-------------------------------------+-------------+
| resnext101\_32x16d   | instagram / ssl / swsl              | 191M        |
+----------------------+-------------------------------------+-------------+
| resnext101\_32x32d   | instagram                           | 466M        |
+----------------------+-------------------------------------+-------------+
| resnext101\_32x48d   | instagram                           | 826M        |
+----------------------+-------------------------------------+-------------+
~~~~~~~~~~
ResNeSt
~~~~~~~~~~

+----------------------------+------------+-------------+
| Encoder                    | Weights    | Params, M   |
+============================+============+=============+
| timm-resnest14d            | imagenet   | 8M          |
+----------------------------+------------+-------------+
| timm-resnest26d            | imagenet   | 15M         |
+----------------------------+------------+-------------+
| timm-resnest50d            | imagenet   | 25M         |
+----------------------------+------------+-------------+
| timm-resnest101e           | imagenet   | 46M         |
+----------------------------+------------+-------------+
| timm-resnest200e           | imagenet   | 68M         |
+----------------------------+------------+-------------+
| timm-resnest269e           | imagenet   | 108M        |
+----------------------------+------------+-------------+
| timm-resnest50d\_4s2x40d   | imagenet   | 28M         |
+----------------------------+------------+-------------+
| timm-resnest50d\_1s4x24d   | imagenet   | 23M         |
+----------------------------+------------+-------------+
~~~~~~~~~~
Res2Ne(X)t
~~~~~~~~~~

+----------------------------+------------+-------------+
| Encoder                    | Weights    | Params, M   |
+============================+============+=============+
| timm-res2net50\_26w\_4s    | imagenet   | 23M         |
+----------------------------+------------+-------------+
| timm-res2net101\_26w\_4s   | imagenet   | 43M         |
+----------------------------+------------+-------------+
| timm-res2net50\_26w\_6s    | imagenet   | 35M         |
+----------------------------+------------+-------------+
| timm-res2net50\_26w\_8s    | imagenet   | 46M         |
+----------------------------+------------+-------------+
| timm-res2net50\_48w\_2s    | imagenet   | 23M         |
+----------------------------+------------+-------------+
| timm-res2net50\_14w\_8s    | imagenet   | 23M         |
+----------------------------+------------+-------------+
| timm-res2next50            | imagenet   | 22M         |
+----------------------------+------------+-------------+
~~~~~~~~~~
RegNet(x/y)
~~~~~~~~~~

+---------------------+------------+-------------+
| Encoder             | Weights    | Params, M   |
+=====================+============+=============+
| timm-regnetx\_002   | imagenet   | 2M          |
+---------------------+------------+-------------+
| timm-regnetx\_004   | imagenet   | 4M          |
+---------------------+------------+-------------+
| timm-regnetx\_006   | imagenet   | 5M          |
+---------------------+------------+-------------+
| timm-regnetx\_008   | imagenet   | 6M          |
+---------------------+------------+-------------+
| timm-regnetx\_016   | imagenet   | 8M          |
+---------------------+------------+-------------+
| timm-regnetx\_032   | imagenet   | 14M         |
+---------------------+------------+-------------+
| timm-regnetx\_040   | imagenet   | 20M         |
+---------------------+------------+-------------+
| timm-regnetx\_064   | imagenet   | 24M         |
+---------------------+------------+-------------+
| timm-regnetx\_080   | imagenet   | 37M         |
+---------------------+------------+-------------+
| timm-regnetx\_120   | imagenet   | 43M         |
+---------------------+------------+-------------+
| timm-regnetx\_160   | imagenet   | 52M         |
+---------------------+------------+-------------+
| timm-regnetx\_320   | imagenet   | 105M        |
+---------------------+------------+-------------+
| timm-regnety\_002   | imagenet   | 2M          |
+---------------------+------------+-------------+
| timm-regnety\_004   | imagenet   | 3M          |
+---------------------+------------+-------------+
| timm-regnety\_006   | imagenet   | 5M          |
+---------------------+------------+-------------+
| timm-regnety\_008   | imagenet   | 5M          |
+---------------------+------------+-------------+
| timm-regnety\_016   | imagenet   | 10M         |
+---------------------+------------+-------------+
| timm-regnety\_032   | imagenet   | 17M         |
+---------------------+------------+-------------+
| timm-regnety\_040   | imagenet   | 19M         |
+---------------------+------------+-------------+
| timm-regnety\_064   | imagenet   | 29M         |
+---------------------+------------+-------------+
| timm-regnety\_080   | imagenet   | 37M         |
+---------------------+------------+-------------+
| timm-regnety\_120   | imagenet   | 49M         |
+---------------------+------------+-------------+
| timm-regnety\_160   | imagenet   | 80M         |
+---------------------+------------+-------------+
| timm-regnety\_320   | imagenet   | 141M        |
+---------------------+------------+-------------+
~~~~~~~~~~
SE-Net
~~~~~~~~~~

+-------------------------+------------+-------------+
| Encoder                 | Weights    | Params, M   |
+=========================+============+=============+
| senet154                | imagenet   | 113M        |
+-------------------------+------------+-------------+
| se\_resnet50            | imagenet   | 26M         |
+-------------------------+------------+-------------+
| se\_resnet101           | imagenet   | 47M         |
+-------------------------+------------+-------------+
| se\_resnet152           | imagenet   | 64M         |
+-------------------------+------------+-------------+
| se\_resnext50\_32x4d    | imagenet   | 25M         |
+-------------------------+------------+-------------+
| se\_resnext101\_32x4d   | imagenet   | 46M         |
+-------------------------+------------+-------------+
~~~~~~~~~~
SK-ResNe(X)t
~~~~~~~~~~

+---------------------------+------------+-------------+
| Encoder                   | Weights    | Params, M   |
+===========================+============+=============+
| timm-skresnet18           | imagenet   | 11M         |
+---------------------------+------------+-------------+
| timm-skresnet34           | imagenet   | 21M         |
+---------------------------+------------+-------------+
| timm-skresnext50\_32x4d   | imagenet   | 25M         |
+---------------------------+------------+-------------+
~~~~~~~~~~
DenseNet
~~~~~~~~~~

+---------------+------------+-------------+
| Encoder       | Weights    | Params, M   |
+===============+============+=============+
| densenet121   | imagenet   | 6M          |
+---------------+------------+-------------+
| densenet169   | imagenet   | 12M         |
+---------------+------------+-------------+
| densenet201   | imagenet   | 18M         |
+---------------+------------+-------------+
| densenet161   | imagenet   | 26M         |
+---------------+------------+-------------+
~~~~~~~~~~
Inception
~~~~~~~~~~

+---------------------+----------------------------------+-------------+
| Encoder             | Weights                          | Params, M   |
+=====================+==================================+=============+
| inceptionresnetv2   | imagenet / imagenet+background   | 54M         |
+---------------------+----------------------------------+-------------+
| inceptionv4         | imagenet / imagenet+background   | 41M         |
+---------------------+----------------------------------+-------------+
| xception            | imagenet                         | 22M         |
+---------------------+----------------------------------+-------------+
~~~~~~~~~~
EfficientNet
~~~~~~~~~~

+------------------------+--------------------------------------+-------------+
| Encoder                | Weights                              | Params, M   |
+========================+======================================+=============+
| efficientnet-b0        | imagenet                             | 4M          |
+------------------------+--------------------------------------+-------------+
| efficientnet-b1        | imagenet                             | 6M          |
+------------------------+--------------------------------------+-------------+
| efficientnet-b2        | imagenet                             | 7M          |
+------------------------+--------------------------------------+-------------+
| efficientnet-b3        | imagenet                             | 10M         |
+------------------------+--------------------------------------+-------------+
| efficientnet-b4        | imagenet                             | 17M         |
+------------------------+--------------------------------------+-------------+
| efficientnet-b5        | imagenet                             | 28M         |
+------------------------+--------------------------------------+-------------+
| efficientnet-b6        | imagenet                             | 40M         |
+------------------------+--------------------------------------+-------------+
| efficientnet-b7        | imagenet                             | 63M         |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b0   | imagenet / advprop / noisy-student   | 4M          |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b1   | imagenet / advprop / noisy-student   | 6M          |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b2   | imagenet / advprop / noisy-student   | 7M          |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b3   | imagenet / advprop / noisy-student   | 10M         |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b4   | imagenet / advprop / noisy-student   | 17M         |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b5   | imagenet / advprop / noisy-student   | 28M         |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b6   | imagenet / advprop / noisy-student   | 40M         |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b7   | imagenet / advprop / noisy-student   | 63M         |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-b8   | imagenet / advprop                   | 84M         |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-l2   | noisy-student                        | 474M        |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-lite0| imagenet                             | 4M          |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-lite1| imagenet                             | 4M          |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-lite2| imagenet                             | 6M          |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-lite3| imagenet                             | 8M          |
+------------------------+--------------------------------------+-------------+
| timm-efficientnet-lite4| imagenet                             | 13M         |
+------------------------+--------------------------------------+-------------+
~~~~~~~~~~
MobileNet
~~~~~~~~~~

+-----------------+------------+-------------+
| Encoder         | Weights    | Params, M   |
+=================+============+=============+
| mobilenet\_v2   | imagenet   | 2M          |
+-----------------+------------+-------------+
~~~~~~~~~~
DPN
~~~~~~~~~~

+-----------+---------------+-------------+
| Encoder   | Weights       | Params, M   |
+===========+===============+=============+
| dpn68     | imagenet      | 11M         |
+-----------+---------------+-------------+
| dpn68b    | imagenet+5k   | 11M         |
+-----------+---------------+-------------+
| dpn92     | imagenet+5k   | 34M         |
+-----------+---------------+-------------+
| dpn98     | imagenet      | 58M         |
+-----------+---------------+-------------+
| dpn107    | imagenet+5k   | 84M         |
+-----------+---------------+-------------+
| dpn131    | imagenet      | 76M         |
+-----------+---------------+-------------+
~~~~~~~~~~
VGG
~~~~~~~~~~

+-------------+------------+-------------+
| Encoder     | Weights    | Params, M   |
+=============+============+=============+
| vgg11       | imagenet   | 9M          |
+-------------+------------+-------------+
| vgg11\_bn   | imagenet   | 9M          |
+-------------+------------+-------------+
| vgg13       | imagenet   | 9M          |
+-------------+------------+-------------+
| vgg13\_bn   | imagenet   | 9M          |
+-------------+------------+-------------+
| vgg16       | imagenet   | 14M         |
+-------------+------------+-------------+
| vgg16\_bn   | imagenet   | 14M         |
+-------------+------------+-------------+
| vgg19       | imagenet   | 20M         |
+-------------+------------+-------------+
| vgg19\_bn   | imagenet   | 20M         |
+-------------+------------+-------------+


# TODO: 
- Update model choices
