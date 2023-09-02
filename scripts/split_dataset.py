import random
import os
import shutil

def move_data(img_dir, mask_dir, set, cate):
    img_dir_ = os.path.join(img_dir, cate)
    mask_dir_ = os.path.join(mask_dir, cate)
    if not os.path.exists(img_dir_):
        os.makedirs(img_dir_)
    if not os.path.exists(mask_dir_):
        os.makedirs(mask_dir_)
    for file in set:
        shutil.move(os.path.join(img_dir, file), img_dir_)
        shutil.move(os.path.join(mask_dir, file), mask_dir_)

if __name__ == "__main__":
    data_dir = '/data1/Invertible_ISP/Invertible_ISP_0.7'
    imgs_dir = f'{data_dir}/imgs'
    masks_dir = f'{data_dir}/masks'
    data = list(os.listdir(imgs_dir))
    random.seed(77)
    random.shuffle(data)
    num = len(data)
    train_rate, val_rate = 0.8, 0.1
    train_num = int(num * train_rate)
    val_num = int(num * val_rate)
    
    train_set = data[:train_num]
    val_set = data[train_num:train_num + val_num]
    test_set = data[train_num + val_num:]
    
    move_data(imgs_dir, masks_dir, train_set, 'train')
    move_data(imgs_dir, masks_dir, val_set, 'val')
    move_data(imgs_dir, masks_dir, test_set, 'test')