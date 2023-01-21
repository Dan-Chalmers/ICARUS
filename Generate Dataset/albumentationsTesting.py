
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
def augment(transform: A.core.composition.Compose, image: NP.ndarray, name: int, shape: str):
    # Augment an image
    for i in range (801):
        transformed = transform(image=image)['image']
        transformed = CV.cvtColor(transformed, CV.COLOR_RGB2BGR)
        CV.imwrite('D://VALIDATION//{0}{1}.png'.format(str(shape), str(name)), transformed)
        name += 1
        transform = getTransform()
    

def main():
    files = ['circle', 'square', 'triangle', 'star', 'pentagon']
    for shape in files:
        img = CV.imread('Shapes//{0}.png'.format(str(shape)))
        img = CV.cvtColor(img, CV.COLOR_RGB2BGR)
        augment(getTransform(), img, 0, shape)


if __name__ == '__main__':
    main()