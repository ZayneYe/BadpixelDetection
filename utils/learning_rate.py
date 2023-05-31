import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from matplotlib import pyplot as plt

class CosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, cosine_annealing, last_epoch=-1, verbose=False):
        self.cosine_annealing = cosine_annealing
        super(CosineAnnealingScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr * self.cosine_annealing(self.last_epoch)
                for base_lr in self.base_lrs]


def cosine_annealing(epoch):
    T_max = 500
    eta_min = 0.01 
    decay_rate = 0.999
    cos_annealing = eta_min + 0.5 * (1 - eta_min) * (1 + torch.cos(torch.tensor(epoch / T_max * np.pi)))
    exp_decay = decay_rate ** epoch
    return cos_annealing * exp_decay