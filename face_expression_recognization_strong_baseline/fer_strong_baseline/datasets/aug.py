import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random as rd

from albumentations import (
    Resize,
    RandomCrop,
    HorizontalFlip,
    GaussNoise,
    Blur,GaussianBlur,MedianBlur,
    HueSaturationValue,
    RandomBrightnessContrast,
    IAASharpen,
    Normalize,
    OneOf, Compose,
    NoOp,
)


class CropByScale(torch.nn.Module):
    def __init__(self, left, top, right, bottom):
        super().__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def forward(self, img):
        h, w, _ = img.shape
        return F.crop(img, h * self.top, w * self.left, h * (self.bottom - self.top), w * (self.right - self.left))

    def __repr__(self):
        return self.__class__.__name__ + '(crop position: left={0}, top={1}, right={2}, bottom={3})'.format(self.left, self.top,
                                                                                                            self.right, self.bottom)


class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """

    def __init__(self, phase):
        assert (phase == 'train' or phase == 'test')
        self.phase = phase

    def __call__(self, img):
        # pre determined parameters
        final_size = 224
        final_width = final_height = final_size
        crop_size = 110
        crop_height = crop_width = crop_size
        crop_center_y_offset = 15
        crop_center_x_offset = 0
        if self.phase == 'train':
            scale_aug = 0.02
            trans_aug = 0.01
        else:
            scale_aug = 0.0
            trans_aug = 0.0

        # computed parameters
        randint = rd.randint
        scale_height_diff = (randint(0, 1000) / 500 - 1) * scale_aug
        crop_height_aug = crop_height * (1 + scale_height_diff)
        scale_width_diff = (randint(0, 1000) / 500 - 1) * scale_aug
        crop_width_aug = crop_width * (1 + scale_width_diff)

        trans_diff_x = (randint(0, 1000) / 500 - 1) * trans_aug
        trans_diff_y = (randint(0, 1000) / 500 - 1) * trans_aug

        center = ((img.width / 2 + crop_center_x_offset) * (1 + trans_diff_x),
                  (img.height / 2 + crop_center_y_offset) * (1 + trans_diff_y))

        if center[0] < crop_width_aug / 2:
            crop_width_aug = center[0] * 2 - 0.5
        if center[1] < crop_height_aug / 2:
            crop_height_aug = center[1] * 2 - 0.5
        if (center[0] + crop_width_aug / 2) >= img.width:
            crop_width_aug = (img.width - center[0]) * 2 - 0.5
        if (center[1] + crop_height_aug / 2) >= img.height:
            crop_height_aug = (img.height - center[1]) * 2 - 0.5

        crop_box = (center[0] - crop_width_aug / 2, center[1] - crop_height_aug / 2,
                    center[0] + crop_width_aug / 2, center[1] + crop_width_aug / 2)

        mid_img = img.crop(crop_box)
        res_img = img.resize((final_width, final_height))
        return res_img

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def  fer_train_aug(input_size, crop_residual_pix = 16):
    aug = Compose(
        [
            # OneOf(
            # [
            #     HorizontalFlip(p = 0.5),
            #     GaussNoise(0, 30)
            # ],
            # p=1.0),
            Resize(height=input_size + crop_residual_pix,
                   width=input_size + crop_residual_pix),
            OneOf(
                [
                    # RandomCrop(height=input_size, width=input_size),
                    Resize(height=input_size,
                        width=input_size)
                 ],
                p=1.0),
            # OneOf(
            #     [
            #         Blur(blur_limit = 7, p = 0.5),
            #         GaussianBlur(blur_limit = 7, p = 0.5),
            #         MedianBlur(blur_limit = 7, p = 0.5)
            #     ]
            # ),
            # OneOf(
            #     [
            #         HueSaturationValue(hue_shift_limit = 30,
            #                            sat_shift_limit = 30,
            #                            val_shift_limit = 30,
            #                            p = 0.5),
            #         RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3)
            #     ]
            # ),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
    p = 1.0)

    return  aug


def  fer_test_aug(input_size):
    aug = Compose(
        [
            Resize(height = input_size,
                   width  = input_size),
            # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ],
    p = 1.0)
    return  aug