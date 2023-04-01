
import albumentations as A
import numpy as NP
import cv2 as CV


''' Declare an augmentation pipeline '''
def getTransform() -> A.core.composition.Compose:
    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.ShiftScaleRotate(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.Blur(),
            A.Defocus(),
            A.GaussianBlur(),
            A.ChannelShuffle(),
            A.PixelDropout(),
            A.RandomFog(),
            A.RandomGamma(),
            A.RandomSunFlare()
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.ShiftScaleRotate(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.Blur(),
            A.Defocus(),
            A.GaussianBlur(),
            A.ChannelShuffle(),
            A.PixelDropout(),
            A.RandomFog(),
            A.RandomGamma(),
            A.RandomSunFlare()
        ]),
    ])

    return transform


''' Apply the augmentation pipeline and save image '''
def augment(transform: A.core.composition.Compose, image: NP.ndarray, name: int):

    # Augment an image
    transformed = transform(image=image)['image']
    transformed = CV.cvtColor(transformed, CV.COLOR_RGB2BGR)
    CV.imwrite('E://tempImageStorage//{0}.png'.format(str(name)), transformed)
    transform = getTransform()