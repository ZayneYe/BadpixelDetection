# Bad Pixel Detection

We propose a binary segmentation method for effective detection of bad pixels. Our approach yields up to 99.6\% detection accuracy with ${<}0.6\%$ false positives. While this approach achieves nearly perfect detection for large datasets, the detection rate drops for smaller datasets. To mitigate this gap, we propose confidence calibration using multiple images during inference. Our confidence-calibrated segmentation approach yields an improvement of up to 20\% over regular binary segmentation. 

[link to paper](https://arxiv.org/pdf/2402.05521.pdf)

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
Multi-gpu: python train.py --device 0,1,2,3

```

Testing 
```
1. Bad Pixel Detection using Segmentation
Single gpu: python test.py

2. Bad Pixel Detection using Confidence Calibration
cd scripts/
python fixed_pattern.py # inject same error pattern in all test images
python vote_test.py
```

## Citation
If you find this repo useful for your research, please consider citing the following work:
```
@InProceedings{sarkar_2024_CVPRW,
    author       = {Sarkar, Sreetama and Ye, Xinan and Datta, Gourav and Beerel, Peter},
    title        = {FixPix: Fixing Bad Pixels using Deep Learning}, 
    eprint       = {2310.11637},
    archivePrefix={arXiv},
    primaryClass ={eess.IV},
    year         = {2024}
}
```
