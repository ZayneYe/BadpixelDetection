import os
import shutil

if __name__ == "__main__":
    download_path = "/data1/fivek_dataset/raw_photos"
    data_path = "/data1/Invertible_ISP/original_images"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    cnt = 0
    for folder in os.listdir(download_path):
        if folder[:3] == "HQa":
            dng_path = os.path.join(download_path, folder, 'photos')
            for dng in os.listdir(dng_path):
                shutil.copyfile(os.path.join(dng_path, dng), os.path.join(data_path, dng))
                cnt += 1
                print(f"{cnt} dng images have been copied.")