import os
import numpy as np
import torch


class SIDDDataset(torch.utils.data.Dataset):
    def __init__(self, dir, train):
        if train:
            self.imgs_path = os.path.join(dir, 'imgs', 'train')
            self.masks_path = os.path.join(dir, 'masks', 'train')
        else:
            self.imgs_path = os.path.join(dir, 'imgs', 'val')
            self.masks_path = os.path.join(dir, 'masks', 'val')

    def __getitem__(self, index):
        img_file = os.listdir(self.imgs_path)[index]
        mask_file = os.listdir(self.masks_path)[index]
        img = np.load(os.path.join(self.imgs_path, img_file))
        mask = np.load(os.path.join(self.masks_path, mask_file))
        return img, mask

    def __len__(self):
        return len(os.listdir(self.imgs_path))