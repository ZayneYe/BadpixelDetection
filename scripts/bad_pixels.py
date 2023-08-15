import os
import numpy as np
import random
import rawpy
import sys

def decide_range(value, delta):
    if (1 + delta) * value > 1023:
        return [(0, int((1 - delta) * value))]
    else:
        return [(0, int((1 - delta) * value)), (int((1 + delta) * value), 1023)]

if __name__ == "__main__":
    data_dir = '../data/ISP_0.7'
    org_dir = f'{data_dir}/original_imgs'
    imgs_dir = f'{data_dir}/imgs'
    masks_dir = f'{data_dir}/masks'
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    delta = 0.7
    bad_rate = 0.0001
    
    for i, dng in enumerate(os.listdir(org_dir)):
        file_name = dng.split('.')[0]
        raw = rawpy.imread(os.path.join(org_dir, dng))
        raw_data = raw.raw_image
        raw_data = np.asarray(raw_data)
        H, W = raw_data.shape
        bad_num = int(bad_rate * H * W)
        random.seed(i)
        bad_pos = [(random.randint(0, H - 1), random.randint(0, W - 1)) for _ in range(bad_num)]
        mask = np.zeros((H, W))
        for x, y in bad_pos:
            o_val = raw_data[x][y]
            ranges = decide_range(o_val, delta)
            chosed_range = random.choice(ranges)
            b_val = random.randint(*chosed_range)
            mask[x][y] = 1
            raw_data[x][y] = b_val
        np.save(os.path.join(imgs_dir, file_name), raw_data)
        np.save(os.path.join(masks_dir, file_name), mask)
        print(f'noisy {dng} is saved.')
            
            
        
    