import numpy as np
import os
import torch

def reconstruct(predict):
    B, _, h, w = predict.shape
    H = (h - 2) * 2
    W = (w - 2) * 2
    mid_H = int(H / 2)
    mid_W = int(W / 2)
    reconstruct =  torch.zeros(B, H, W)
    reconstruct[:, :mid_H,:mid_W] = predict[:, 0, :mid_H,:mid_W]
    reconstruct[:, mid_H:H,:mid_W] = predict[:, 1, :mid_H,:mid_W]
    reconstruct[:, :mid_H,mid_W:W] = predict[:, 2, :mid_H,:mid_W]
    reconstruct[:, mid_H:H,mid_W:W] = predict[:, 3, :mid_H,:mid_W]
    return reconstruct

def divide4(fea, idx):
    H, W = fea.shape
    
    fea1 = fea[0:int(H/2 + 2), 0:int(W/2 + 2)]
    fea2 = fea[int(H/2 - 2):H, 0:int(W/2 + 2)]
    fea3 = fea[0:int(H/2 + 2), int(W/2 - 2):W]
    fea4 = fea[int(H/2 - 2):H, int(W/2 - 2):W]
    fea_combine = np.stack((fea1, fea2, fea3, fea4))
    return fea_combine[idx]

def prepocess(img, mask, idx):
    img_ = divide4(img, idx)
    mask_ = divide4(mask, idx)
    return img_, mask_


if __name__ == '__main__':
    npy_dir = "/home/xinanye/project/Badpixels/data/SIDD_DNG/masks"
    for cate in os.listdir(npy_dir):
        for npy in os.listdir(os.path.join(npy_dir, cate)):
            npy = np.load(os.path.join(npy_dir, cate, npy))
            npy_4 = divide4(npy)
            npy_re = reconstruct(npy_4)
            print(npy_re.all() == npy.all())