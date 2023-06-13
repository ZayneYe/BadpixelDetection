from matplotlib import pyplot as plt
import os
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

def plot_loss_curve(loss_vec, val_vec, val_loss_vec, save_path):
    plt.figure(0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss_vec)
    plt.plot(val_vec, val_loss_vec)
    plt.legend(labels=["Training", "Validation"], loc="upper right", fontsize=12)
    plt.savefig(os.path.join(save_path, 'Loss_curve.png'))

def plot_NMSE(img_size, test_loss_vec, mean_loss_vec, median_loss_vec, save_path):
    plt.figure(1)
    plt.title(f"Prediction using {img_size}x{img_size} patches")
    plt.xlabel('Number of corrupted pixels')
    plt.ylabel('Test NMSE')
    plt.plot(test_loss_vec, label='MLP')
    plt.plot(mean_loss_vec, label='Mean')
    plt.plot(median_loss_vec, label='Median')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_path, 'Test_NMSE.png'))

def plot_multisize_NMSE(nmses_dict, save_path):
    nmses_vec = sorted(nmses_dict.items(), key=lambda x:x[0])
    nmses_dict = dict(nmses_vec)
    plt.figure(2, figsize=(8,6))
    plt.title(f"Comparison of MLPs using different patches", size=18)
    plt.xlabel('Number of corrupted pixels', size=18)
    plt.ylabel('Test NMSE', size=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    for patch_size in nmses_dict:
        plt.plot(nmses_dict[patch_size], label=f'{patch_size}x{patch_size} patches')
        plt.scatter(range(len(nmses_dict[patch_size])), nmses_dict[patch_size], marker='o')
    plt.legend(loc="upper left",  prop ={'size':15})
    plt.grid()
    plt.savefig(os.path.join(save_path, 'Multisize_NMSE.png'))

def plot_mean_median(cate_vec, loss_vec, save_path):
    plt.figure(3)
    plt.xlabel('Predict Method')
    plt.ylabel('Test NMSE')
    for x, y in zip(cate_vec, loss_vec):
        plt.text(x, y, '%.4f' % y, ha='center', va='bottom')
    plt.bar(cate_vec, loss_vec, width=0.25)
    plt.savefig(os.path.join(save_path, 'Test_NMSE.png'))

def plot_multimodel_NMSE(nmses_dict, save_path):
    df = pd.DataFrame(nmses_dict)
    df.to_excel(os.path.join(save_path, 'nmse.xlsx'), index=False, engine='openpyxl')
    plt.figure(4, figsize=(8,6))
    plt.title("Comparison of MLPs trained by different data", size=18)
    plt.xlabel('Number of corrupted pixels', size=18)
    plt.ylabel('Test NMSE', size=18)
    xticks = range(len(nmses_dict['model_1']))
    plt.xticks(xticks, size=18)
    plt.yscale('log')
    plt.yticks(size=18)
    plt.grid()
    color_vec = ['r', 'b', 'y', 'c', 'm', 'k', 'g',]
    for i, model in enumerate(nmses_dict):
        if model == 'model_dist':
            continue
        model_order = model.split('_')[1]
        x_vec = range(len(nmses_dict[model]))
        y_vec = nmses_dict[model]
        if model_order == '0' or model_order == '1':
            legend = f'Trained with {model_order} corrupted pixel'
        else:
            legend = f'Trained with {model_order} corrupted pixels'
        plt.plot(x_vec, y_vec, label=legend, color=color_vec[i])
        plt.scatter(x_vec, y_vec, color=color_vec[i], marker='o')
        # for x, y in zip(x_vec, y_vec):
        #     plt.text(x, y, '%.4f' % y, ha='center', va='center', color=color_vec[i])
    
    plt.legend(loc="upper left", prop ={'size':15})
    plt.savefig(os.path.join(save_path, 'Multimodel_NMSE.png'))
    

def plot_prcurve(r_vec, p_vec, iou_vec, dice_vec, save_path):
    cof_vec = np.linspace(0,1,31)
    plt.figure(5, figsize=(8,6))
    plt.xlabel('Recall', size=18)
    plt.ylabel('Precision', size=18)
    plt.plot(r_vec, p_vec)
    plt.savefig(os.path.join(save_path, 'PR_curve.png'))
    plt.figure(6, figsize=(8,6))
    plt.xlabel('Confidence', size=18)
    plt.ylabel('Recall', size=18)
    plt.plot(cof_vec, r_vec)
    plt.savefig(os.path.join(save_path, 'R_curve.png'))
    plt.figure(7, figsize=(8,6))
    plt.xlabel('Confidence', size=18)
    plt.ylabel('Precision', size=18)
    plt.plot(cof_vec, p_vec)
    plt.savefig(os.path.join(save_path, 'P_curve.png'))
    plt.figure(8, figsize=(8,6))
    plt.xlabel('Confidence', size=18)
    plt.ylabel('IOU', size=18)
    plt.plot(cof_vec, iou_vec)
    plt.savefig(os.path.join(save_path, 'IOU_curve.png'))
    plt.figure(9, figsize=(8,6))
    plt.xlabel('Confidence', size=18)
    plt.ylabel('Dice', size=18)
    plt.plot(cof_vec, dice_vec)
    plt.savefig(os.path.join(save_path, 'Dice_curve.png'))


def plot_lr_curve(lr_vec, save_path):
    plt.figure(10)
    plt.xlabel('iterations', size=18)
    plt.ylabel('lr', size=18)
    plt.plot(lr_vec)
    plt.savefig(os.path.join(save_path, 'lr_curve.png'))


def plot_iou_curve(iou_vec, dice_vec, val_vec, save_path):
    plt.figure(11, figsize=(8,6))
    plt.xlabel('Epochs', size=18)
    plt.ylabel('IOU', size=18)
    plt.plot(val_vec, iou_vec)
    plt.savefig(os.path.join(save_path, 'IOU_learn_curve.png'))
    plt.figure(12, figsize=(8,6))
    plt.xlabel('Epochs', size=18)
    plt.ylabel('Dice', size=18)
    plt.plot(val_vec, dice_vec)
    plt.savefig(os.path.join(save_path, 'Dice_learn_curve.png'))