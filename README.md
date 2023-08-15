# Badpixels
 
This is the baseline network for detect the bad pixels location using segmentation method for the Samsung In-Sensor Computing project.

The folder hierarchy for the dataset should look like this:

<img width="95" alt="1678766420624" src="https://user-images.githubusercontent.com/106359260/224889848-bf2e552b-e403-42a9-8892-2e8fe63519e8.png">


where the S7-ISP-Dataset is download from https://www.kaggle.com/datasets/knn165897/s7-isp-dataset.

To train:
python train.py

To evaluate:
python test.py --mode test
