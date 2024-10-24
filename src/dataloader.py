import torchvision.transforms.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np

import os
import random

from PIL import Image, ImageOps

# import any other libraries you need below this line
import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2.functional import adjust_gamma


class Cell_data(Dataset):
    def __init__(self, data_dir, size, train=True, train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # initialize the data class
        self.data_dir = data_dir
        self.size = size
        self.train = train
        self.train_test_split = train_test_split
        self.augment_data = augment_data

        # read images and masks
        image_path = os.path.join(data_dir, 'scans')
        mask_path = os.path.join(data_dir, 'labels')

        image_files = sorted(
            [os.path.join(image_path, file) for file in os.listdir(image_path) if file.endswith('.bmp')])
        masks = sorted([os.path.join(mask_path, file) for file in os.listdir(mask_path) if file.endswith('.bmp')])

        # split train set & test set
        idx = int(train_test_split * len(image_files))
        if train:
            self.images = image_files[:idx]
            self.masks = masks[:idx]
        else:
            self.images = image_files[idx:]
            self.masks = masks[idx:]

    def __getitem__(self, idx):

        # load image and mask from index idx of your data
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image_PIL = Image.open(image_path).resize((self.size, self.size))
        mask_PIL = Image.open(mask_path).resize((self.size, self.size))

        if not self.train:
            return self.load_image(image_PIL), self.load_mask(mask_PIL)

        # data augmentation part
        # reference: https://pytorch.org/vision/main/auto_examples/transforms
        # /plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py

        if self.augment_data:
            augment_mode = np.random.randint(0, 6)
            if augment_mode == 0:
                # flip image vertically
                image_PIL = TF.vflip(image_PIL)
                mask_PIL = TF.vflip(mask_PIL)

            elif augment_mode == 1:
                # flip image horizontally
                image_PIL = TF.hflip(image_PIL)
                mask_PIL = TF.hflip(mask_PIL)

            elif augment_mode == 2:
                # zoom image
                output_size = (self.size, self.size)
                resize_transform = v2.RandomResizedCrop(size=output_size)
                params = resize_transform.get_params(image_PIL, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))

                image_PIL = TF.resized_crop(image_PIL, *params, size=output_size)
                mask_PIL = TF.resized_crop(mask_PIL, *params, size=output_size)

            elif augment_mode == 3:
                # rotate image
                angle = random.uniform(0, 360)
                image_PIL = TF.rotate(image_PIL, angle)
                mask_PIL = TF.rotate(mask_PIL, angle)

            elif augment_mode == 4:
                # Apply ElasticTransform
                # Based on the original paper, alpha = 10, sigma = 10
                elastic_transform = v2.ElasticTransform(alpha=10.0, sigma=10.0)
                image_PIL = elastic_transform(image_PIL)
                mask_PIL = elastic_transform(mask_PIL)

            else:
                # Gamma correction
                # reference: https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_gamma.html
                gamma = random.uniform(0.5, 1.5)
                image_PIL = adjust_gamma(image_PIL, gamma=gamma)

        # return image and mask in tensors
        image_tensor = self.load_image(image_PIL)
        mask = self.load_mask(mask_PIL)
        return image_tensor, mask

    # Helper function to load images, given file, return a tensor
    def load_image(self, file):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(file)
        return image_tensor

    # Helper function to load masks, given file, return a tensor
    def load_mask(self, file):
        mask_tensor = torch.from_numpy(np.array(file)).long()  # convert to tensor
        return mask_tensor

    def __len__(self):
        # print(f"Train: {self.train}, length: {len(self.images)}")
        return len(self.images)
