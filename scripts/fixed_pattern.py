import os
import shutil
import rawpy
import numpy as np
import random

def generate_pattern(height, width, bad_rate, idx=7):
    nums_select = int(bad_rate * height * width)
    np.random.seed(idx)
    row_coords = np.random.randint(0, height, (nums_select, 1))
    col_coords = np.random.randint(0, width, (nums_select, 1))
    select_coords = np.hstack((row_coords, col_coords))
    return select_coords

def decide_range(value, delta):
    if (1 + delta) * value > 1023:
        return [(0, int((1 - delta) * value))]
    else:
        return [(0, int((1 - delta) * value)), (int((1 + delta) * value), 1023)]
    
bad_rate = 0.0001
height, width = 3024, 4032
delta = 0.7

test_dir = "/data1/Bad_Pixel_Detection/data/ISP_0.7/imgs/test"
fix_dir = os.path.join(test_dir.split("test")[0], "fixed_pattern")
# org_dir = os.path.join(test_dir.split("imgs")[0], "original_imgs")
org_dir = '/data1/S7-ISP-Dataset/medium_dng'
mask_dir = os.path.join(test_dir.split("imgs")[0], "masks/fixed_pattern")

if not os.path.exists(fix_dir):
    os.makedirs(fix_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
select_coords = generate_pattern(height, width, bad_rate)
pixel_var = []
for c in select_coords:
    range = [(0, (1 - delta)), ((1 + delta), 5.0)] # limit max deviation to 100%
    chosen_range = random.choice(range) # choose +delta or -delta
    chosen_delta = random.uniform(*chosen_range)
    # chosen_delta = random.choice([1-delta,1+delta])
    pixel_var.append(chosen_delta)
for npy in os.listdir(test_dir):
    dng = f"{npy.split('.')[0]}.dng"
    # print(dng)
    raw = rawpy.imread(os.path.join(org_dir, dng))
    raw_data = raw.raw_image
    raw_data = np.asarray(raw_data)
    H, W = raw_data.shape
    mask = np.zeros((H, W))
    for i,c in enumerate(select_coords):
        o_val = raw_data[c[0]][c[1]]
        # ranges = decide_range(o_val, delta)
        # chosed_range = random.choice(ranges)
        # b_val = random.randint(*chosed_range)
        b_val = min(max(pixel_var[i] * o_val, 0), 1023)
        print(o_val, b_val)
        mask[c[0]][c[1]] = 1
        raw_data[c[0]][c[1]] = b_val
    np.save(os.path.join(fix_dir, npy), raw_data)
    np.save(os.path.join(mask_dir, npy), mask)
    # shutil.copyfile(os.path.join(org_dir, dng), os.path.join(fix_dir, dng))
        
