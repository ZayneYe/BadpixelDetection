import os
import rawpy

data_path = "/data1/Invertible_ISP/original_images"

maxH, maxW = 0, 0
cnt = {}
for dng in os.listdir(data_path):
    raw = rawpy.imread(os.path.join(data_path, dng))
    raw_data = raw.raw_image
    if len(raw_data.shape) > 2:
        continue
    H, W = raw_data.shape
    if (H,W) in cnt:
        cnt[(H,W)] += 1
    else:
        cnt[(H,W)] = 1
    maxH = max(maxH, H)
    maxW = max(maxW, W)

    print(cnt, maxH, maxW)