import numpy as np
from utils.process import reconstruct
import torch

def binary_iou(pred, lab):
    intersecion = torch.multiply(pred, lab)
    union = pred + lab
    iou = torch.sum(intersecion) / torch.sum(union)
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
    
    return cm, recall, precision, iou, tn, fp, fn, tp
                   
