import numpy as np
from utils.process import reconstruct
import torch
from torch import Tensor

def binary_iou(pred, lab):
    intersecion = pred * lab
    union = pred + lab
    iou = intersecion.sum() / union.sum()
    return iou

def precision_recall(tp, fn, fp):
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return recall, precision

def confuse_matrix(pred, lab):
    TP = torch.sum((lab == 1) & (pred == 1))
    FP = torch.sum((lab == 0) & (pred == 1))
    TN = torch.sum((lab == 0) & (pred == 0))
    FN = torch.sum((lab == 1) & (pred == 0))
    cm = torch.tensor([[TN, FP], [FN, TP]])
    return cm, TN, FP, FN, TP
    
def calc_metrics(predict, label, thres):
    # pred_recon = reconstruct(predict)
    # lab_recon = reconstruct(label)
    pred_binary = (predict >= thres)
    iou = binary_iou(pred_binary, label)
    cm, tn, fp, fn, tp = confuse_matrix(pred_binary, label)
    recall, precision = precision_recall(tp, fn, fp)
    dice_score = dice_coeff(pred_binary, label)
    return recall, precision, iou, dice_score, cm

def calc_metrics_one(predict, label, thres):
    fp_, fn_, tp_, dice_score = 0, 0, 0, 0
    for i in range(len(predict)):
        pred = predict[i]
        lab = label[i]
        pred_binary = (pred >= thres)
        cm, tn, fp, fn, tp = confuse_matrix(pred_binary, lab)
        tp_ += tp
        fn_ += fn
        fp_ += fp
        dice = dice_coeff(pred, lab)
        dice_score += dice
    recall, precision = precision_recall(tp_, fn_, fp_)
    iou = tp_ / (tp_ + fn_ + fp_)
    dice_score /= len(predict)
    return recall, precision, iou, dice_score
        
                
def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-10):
    assert input.size() == target.size()
    # assert input.dim() == 3 or not reduce_batch_first
    sum_dim = (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input: Tensor, target: Tensor):
    return 1 - dice_coeff(input, target)