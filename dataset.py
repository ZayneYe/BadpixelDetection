import os
import numpy as np
import torch
from utils.prepocess import prepocess

MAXV = 1023

class SIDDDataset(torch.utils.data.Dataset):
    def __init__(self, dir, train, transform=None, divide=None):
        self.transform = transform
        self.divide = divide
        if train:
            self.imgs_path = os.path.join(dir, 'imgs', 'train')
            self.masks_path = os.path.join(dir, 'masks', 'train')
        else:
            self.imgs_path = os.path.join(dir, 'imgs', 'val')
            self.masks_path = os.path.join(dir, 'masks', 'val')

    def __getitem__(self, index):
        img_file = os.listdir(self.imgs_path)[index]
        mask_file = os.listdir(self.masks_path)[index]
        img = np.load(os.path.join(self.imgs_path, img_file)).astype(np.float32) / MAXV
        mask = np.load(os.path.join(self.masks_path, mask_file))
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        if self.divide:
            img, mask = prepocess(img, mask)
            # print("\n")
        return img, mask

    def __len__(self):
        return len(os.listdir(self.imgs_path))