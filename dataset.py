import os
import numpy as np
import torch
from utils.process import preprocess
import torch.nn.functional as F
from torchvision import transforms


class SamsungDataset(torch.utils.data.Dataset):
    def __init__(self, dir, cate, transform=None, mask_transform=None, patch_num=4):
        self.transform = transform
        self.mask_transform = mask_transform
        self.patch_num = patch_num
        self.imgs_path = os.path.join(dir, 'imgs', cate)
        self.masks_path = os.path.join(dir, 'masks', cate)
        

    def __getitem__(self, index):
        idx = index % self.patch_num
        file = os.listdir(self.imgs_path)[index // self.patch_num]
        # mask_file = os.listdir(self.masks_path)[index // self.patch_num]
        img = np.load(os.path.join(self.imgs_path, file)).astype(np.float32)
        mask = np.load(os.path.join(self.masks_path, file)).astype(np.float32)
       
        if self.patch_num != 1:
            img, mask = preprocess(img, mask, idx, self.patch_num)
        
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask, file

    def __len__(self):
        return self.patch_num * len(os.listdir(self.imgs_path))