import os
import numpy as np
import torch
from utils.process import prepocess
import torch.nn.functional as F
from torchvision import transforms


class SamsungDataset(torch.utils.data.Dataset):
    def __init__(self, dir, train, transform=None, mask_transform=None, patch_num=4):
        self.transform = transform
        self.mask_transform = mask_transform
        self.patch_num = patch_num
        if train:
            self.imgs_path = os.path.join(dir, 'imgs', 'train')
            self.masks_path = os.path.join(dir, 'masks', 'train')
        else:
            self.imgs_path = os.path.join(dir, 'imgs', 'val')
            self.masks_path = os.path.join(dir, 'masks', 'val')

    def __getitem__(self, index):
        idx = index % self.patch_num
        img_file = os.listdir(self.imgs_path)[index // self.patch_num]
        mask_file = os.listdir(self.masks_path)[index // self.patch_num]
        img = np.load(os.path.join(self.imgs_path, img_file)).astype(np.float32)
        mask = np.load(os.path.join(self.masks_path, mask_file)).astype(np.float32)
       
        if self.patch_num != 1:
            img, mask = prepocess(img, mask, idx)
        
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return img, mask

    def __len__(self):
        return self.patch_num * len(os.listdir(self.imgs_path))