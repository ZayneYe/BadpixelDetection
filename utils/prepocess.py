import numpy as np

def divide4(fea):
    _, H, W = fea.shape
    fea1 = fea[:, 0:int(H/2 + 2), 0:int(W/2 + 2)]
    fea2 = fea[:, int(H/2 - 2):H, 0:int(W/2 + 2)]
    fea3 = fea[:, 0:int(H/2 + 2), int(W/2 - 2):W]
    fea4 = fea[:, int(H/2 - 2):H, int(W/2 - 2):W]
    # print(fea1.shape)
    # print(fea2.shape)
    # print(fea3.shape)
    # print(fea4.shape)
    
    return np.concatenate((fea1, fea2, fea3, fea4), axis=0)

def prepocess(img, mask):
    img_ = divide4(img)
    mask_ = divide4(mask)
    return img_, mask_
