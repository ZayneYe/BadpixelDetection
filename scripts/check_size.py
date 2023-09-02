import os
import rawpy

data_path = "/data1/Invertible_ISP/original_images"

for dng in os.listdir(data_path):
    raw = rawpy.imread(os.path.join(data_path, dng))
    raw_data = raw.raw_image
    print(raw_data.shape)