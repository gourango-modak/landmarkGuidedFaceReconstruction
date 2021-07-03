import sys
import os
import random
import glob
import torch
from skimage import io
from skimage import transform as ski_transform
from skimage.color import rgb2gray
import scipy.io as sio
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Lambda, Compose
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
from .utils.utils import cv_crop, cv_rotate, draw_gaussian, transform, power_transform, shuffle_lr, fig2data, generate_weight_map
from PIL import Image
import cv2
import copy
import math
from imgaug import augmenters as iaa


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            image_small = np.expand_dims(image_small, axis=2)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float().div(255.0)}
    

class Preprocess_images():
    def __init__(self, num_landmarks=68, gray_scale=False,
                 detect_face=False, enhance=False, center_shift=0,
                 transform=None,):
        """
        Args:
            landmark_dir (string): Path to the mat file with landmarks saved.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_lanmdkars = num_landmarks
        self.transform = transform
        self.gray_scale = gray_scale
        self.detect_face = detect_face
        self.enhance = enhance
        self.center_shift = center_shift
        if self.detect_face:
            self.face_detector = MTCNN(thresh=[0.5, 0.6, 0.7])

    
    def load_img(self, img_path):
        # img_name = img_path
        # pil_image = Image.open(img_name)
        # if pil_image.mode != "RGB":
        #     # if input is grayscale image, convert it to 3 channel image
        #     if self.enhance:
        #         pil_image = power_transform(pil_image, 0.5)
        #     temp_image = Image.new('RGB', pil_image.size)
        #     temp_image.paste(pil_image)
        #     pil_image = temp_image
        # image = np.array(pil_image)
        image = cv2.resize(img_path, (256, 256))
        if self.gray_scale:
            image = rgb2gray(image)
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=2)
            image = image * 255.0
            image = image.astype(np.uint8)
        if not self.detect_face:
            center = [450//2, 450//2+0]
            if self.center_shift != 0:
                center[0] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
            scale = 1.8
        else:
            detected_faces = self.face_detector.detect_image(image)
            if len(detected_faces) > 0:
                box = detected_faces[0]
                left, top, right, bottom, _ = box
                center = [right - (right - left) / 2.0,
                        bottom - (bottom - top) / 2.0]
                center[1] = center[1] - (bottom - top) * 0.12
                scale = (right - left + bottom - top) / 195.0
            else:
                center = [450//2, 450//2+0]
                scale = 1.8
            if self.center_shift != 0:
                shift = self.center * self.center_shift / 450
                center[0] += int(np.random.uniform(-shift, shift))
                center[1] += int(np.random.uniform(-shift, shift))
        
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        # print(sample['image'].shape)
        sample['image'] = torch.unsqueeze(sample['image'], 1)
        sample['image'] = sample['image'].permute(1,0,2,3)
        return sample


def get_image(image,
                num_landmarks=68, rotation=0, scale=0,
                center_shift=0, random_flip=False,
                brightness=0, contrast=0, saturation=0,
                blur=False, noise=False, jpeg_effect=False,
                random_occlusion=False, gray_scale=False,
                detect_face=False, enhance=False):

    pre_processor = Preprocess_images(num_landmarks=num_landmarks,
                                       gray_scale=gray_scale,
                                       detect_face=detect_face,
                                       enhance=enhance,
                                       transform=ToTensor())
    image = pre_processor.load_img(image)
    return image
