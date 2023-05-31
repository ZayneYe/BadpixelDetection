import os
import numpy as np
import rawpy


data_dir = '/home/xinanye/project/Badpixels/data/ISP/original_imgs'
raw_sum = np.zeros((3024, 4032))
for dng in os.listdir(data_dir):
    raw = rawpy.imread(os.path.join(data_dir, dng))
    raw_data = raw.raw_image
    raw_sum += raw_data
raw_sum /= len(os.listdir(data_dir))

print(np.mean(raw_sum))
print(np.std(raw_sum))
