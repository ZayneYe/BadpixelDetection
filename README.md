# Bad Pixel Detection

This code is for detection of bad pixels in Bayer iamges using Binary Segmentation

Dataset Preparation
```
1. Download Dataset
Samsung S7 ISP Dataset: https://www.kaggle.com/datasets/knn165897/s7-isp-dataset

2. Extract .dng images with medium exposure
cd scripts/
python create_dataset.py

3. Bad Pixel Injection
python bad_pixels.py

4. Split into train, validation and test sets
python split_dataset.py 
``` 

Training
```
Single gpu: python train.py
Multi-gpu: python train.py --device 0 1 2 3

```

Testing 
```
1. Bad Pixel Detection using Segmentation
Single gpu: python test.py
Multi-gpu: python test.py --device 0 1 2 3

2. Bad Pixel Detection using Confidence Calibration
cd scripts/
python fixed_pattern.py # inject same error pattern in all test images
python vote_test.py
```
