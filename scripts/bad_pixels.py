import os
import numpy as np
import random
import rawpy
import sys

def generate_bad_pixel(value, delta, max_pixel_value = 1023):
    if (1 + delta) * value > max_pixel_value:
        delta_range = [(0, int((1 - delta) * value))]
    else:
        delta_range = [(0, int((1 - delta) * value)), (int((1 + delta) * value), max_pixel_value)]
    return random.randint(*random.choice(delta_range))

def decide_range(value, delta):
    if (1 + delta) * value > 1023:
        return [(0, int((1 - delta) * value))]
    else:
        return [(0, int((1 - delta) * value)), (int((1 + delta) * value), 1023)]

if __name__ == "__main__":
    delta = 0.7
    bad_rate = 0.85
    # data_dir = f'/data1/Bad_Pixel_Detection/data/FiveK_Canon_{delta}_{bad_rate}'
    # org_dir = '/data1/FiveK/Canon_EOS_5D/DNG'
    max_pixel_value = 1023 # Samsung S7 ISP
    # max_pixel_value = 4095 # FiveK Canon
    # max_pixel_value = 16383 # FiveK Nikon
    data_dir = f'/data1/Bad_Pixel_Detection/data/ISP_{delta}_{bad_rate}'
    org_dir = '/data1/S7-ISP-Dataset/medium_dng'

    imgs_dir = f'{data_dir}/imgs'
    masks_dir = f'{data_dir}/masks'
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    
    
    err, crt = 0, 0
    for i, dng in enumerate(os.listdir(org_dir)):
        file_name = dng.split('.')[0]
        raw = rawpy.imread(os.path.join(org_dir, dng))
        raw_data = raw.raw_image
        raw_data = np.asarray(raw_data)
        if len(raw_data.shape) != 2:
            err += 1
            print(f"DNG size: {raw_data.shape}, Error num: {err}")
            continue
        else:
            crt += 1
        H, W = raw_data.shape
        bad_num = int(bad_rate * H * W)
        random.seed(i)
        
        # bad_pos = [(random.randint(0, H - 1), random.randint(0, W - 1)) for _ in range(bad_num)]
        mask = np.zeros((H, W))
        
        # Calculate the total number of pixels in the image
        total_pixels = H * W

        # Calculate the number of pixels to modify (70% of total_pixels)
        pixels_to_modify = int(bad_rate * total_pixels)
        
        # Randomly select pixel indices to modify
        pixel_indices = np.random.choice(total_pixels, size=pixels_to_modify, replace=False)

        # Split the pixel indices into x and y coordinates
        y_indices, x_indices = np.unravel_index(pixel_indices, (H, W))

        # Get the current pixel values for the selected pixels
        current_pixel_values = raw_data[y_indices, x_indices]

        # Generate bad pixel values based on the current pixel values using the function
        bad_pixel_values = np.vectorize(generate_bad_pixel)(current_pixel_values, delta, max_pixel_value)

        # Clip the bad pixel values to ensure they are within the valid range (0-255)
        # bad_pixel_values = np.clip(bad_pixel_values, 0, 255)

        # Apply the generated bad pixel values to the selected pixels
        raw_data[y_indices, x_indices] = bad_pixel_values
        mask[y_indices, x_indices] = 1

        # for x, y in bad_pos:
        #     o_val = raw_data[x][y]
        #     ranges = decide_range(o_val, delta)
        #     chosed_range = random.choice(ranges)
        #     b_val = random.randint(*chosed_range)
        #     mask[x][y] = 1
        #     raw_data[x][y] = b_val
        np.save(os.path.join(imgs_dir, file_name), raw_data)
        np.save(os.path.join(masks_dir, file_name), mask)
        print(f'noisy {dng} is saved.')
    print(f"Bad pixels injection finished. {crt} success, {err} fail.")      
            
