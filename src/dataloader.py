import cv2
import torch
import torchvision.transforms.transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

# import any other libraries you need below this line
import torchvision.transforms.functional as v1
import torchvision.transforms.v2 as v2


class Cell_data(Dataset):
    def __init__(self, data_dir, size, train='True', train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class
        self.data_dir = data_dir
        self.size = size
        self.train = train
        self.train_test_split = train_test_split
        self.augment_data = augment_data

        # read images and masks
        image_path = os.path.join(data_dir, 'scans')
        mask_path = os.path.join(data_dir, 'labels')
        a = os.listdir(image_path)
        images = sorted([os.path.join(image_path, file) for file in os.listdir(image_path) if file.endswith('.bmp')])
        masks = sorted([os.path.join(mask_path, file) for file in os.listdir(mask_path) if file.endswith('.bmp')])

        # split train set & test set
        idx = int(train_test_split * len(images))
        if train:
            self.images = images[:idx]
            self.masks = masks[:idx]
        else:
            self.images = images[idx:]
            self.masks = masks[idx:]

    def __getitem__(self, idx):

        # load image and mask from index idx of your data
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)

        if not self.train:
            return image, mask

        # data augmentation part
        # reference: https://pytorch.org/vision/main/auto_examples/transforms
        # /plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
        if self.augment_data:
            augment_mode = np.random.randint(0, 5)
            if augment_mode == 0:
                print("flip vertically")
                # flip image vertically
                image = v1.vflip(image)
                mask = v1.vflip(mask)
            elif augment_mode == 1:
                print("flip horizontally")
                # flip image horizontally
                image = v1.hflip(image)
                mask = v1.hflip(mask)
            elif augment_mode == 2:
                print("zoom image")
                # zoom image
                image = v2.RandomResizedCrop(size=(self.size, self.size))(image)
                mask = v2.RandomResizedCrop(size=(self.size, self.size))(image)
            elif augment_mode == 3:
                print("rotate image")
                # rotate image
                image = v2.RandomRotation((0, 360))(image)
                mask = v2.RandomRotation((0, 360))(mask)
            elif augment_mode == 4:
                print("non-rigid transformation")
                # Convert image and mask tensors to PIL images
                image_pil = v1.to_pil_image(image)
                mask_pil = v1.to_pil_image(mask)

                # Apply ElasticTransform
                elastic_transform = v2.ElasticTransform(alpha=50.0)
                image_pil = elastic_transform(image_pil)
                mask_pil = elastic_transform(mask_pil)

                # Convert back to tensors
                image = v1.to_tensor(image_pil)
                mask = v1.to_tensor(mask_pil)
            else:
                print("gamma correction")

        # todo
        # return image and mask in tensors
        return image, mask

    def __len__(self):
        return len(self.images)

    # Helper function to load images, given file path, return a tensor
    def load_image(self, path):
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (self.size, self.size))
        image = cv2.normalize(image, image, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # convert to tensor
        return image_tensor

    # Helper function to load masks, given file path, return a tensor
    def load_mask(self, path):
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (self.size, self.size))
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # convert to tensor
        return image_tensor
