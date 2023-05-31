import argparse
import os
import sys
import torch
from tqdm import tqdm
from dataset import SamsungDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model import UNet
from utils.logger import get_logger
from utils.plot import plot_loss_curve, plot_prcurve, plot_lr_curve
from utils.metrics import calc_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class PixelCalculate():
    def __init__(self, args):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([151.78], [48.85])])
        mask_transform = transforms.Compose([transforms.ToTensor()])
        train_data = SamsungDataset(args.data_path, train=True, transform=transform, mask_transform=mask_transform)
        val_data = SamsungDataset(args.data_path, train=False, transform=transform, mask_transform=mask_transform)
            
        self.train_set = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        self.val_set = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.lr = args.lr
        self.epochs = args.epochs
        self.val_step = args.val_step
        self.model = UNet(1, 1).to(self.device)
        self.model_path = os.path.join("runs/train", args.model_path)
        self.cls_thres = args.cls_thres
        self.criterion = torch.nn.BCELoss()
        self.criterion.to(self.device)
        idx = 0
        exp_dir = 'exp'
        if not os.path.exists(self.model_path):
            self.save_path = os.path.join(self.model_path, 'exp')
        else:
            while(os.path.exists(os.path.join(self.model_path, exp_dir))):
                idx += 1
                exp_dir = f'exp{idx}'
            self.save_path = os.path.join(self.model_path, exp_dir)
        # self.save_path = os.path.join(self.save_path, 'train')
        self.weights_path = os.path.join(self.save_path, 'weights')
        os.makedirs(self.weights_path)

        self.logger = get_logger(os.path.join(self.save_path, f'{exp_dir}_train.log'))
        self.logger.info(vars(args))
        
        
    def save_model(self, save_path, file_name):
        f = os.path.join(save_path, file_name + ".pt")
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module, f)
        else:
            torch.save(self.model, f)


    def validate(self, thres):
        self.model.eval()
        val_loss = 0
        tp_, fp_, fn_ = 0, 0, 0
        with torch.no_grad():
            with tqdm(total=len(self.val_set), desc=f'Eval', unit='batch') as pbar:
                for i, (feature, label) in enumerate(self.val_set):
                    input, target = feature.to(self.device), label.to(self.device)
                    predict = self.model(input)
                    loss = self.criterion(predict, target)
                    val_loss += loss.item()
                    cm, r, p, iou, tn, fp, fn, tp = calc_metrics(predict, label, thres)
                    # print(cm)
                    tp_ += tp
                    fp_ += fp
                    fn_ += fn
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
        precision = tp_ / (tp_ + fp_ + 1e-10)
        recall = tp_ / (tp_ + fn_ + 1e-10)
        IOU = tp_ / (tp_ + fn_ + fp_ + 1e-10)
        val_loss /= len(self.val_set)
        return val_loss, recall, precision, IOU


    def train(self):
        self.logger.info("Start Training...")
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        
        train_loss, val_loss, min_val_loss = 0, 0, sys.maxsize
        loss_vec, val_loss_vec, val_vec, lr_vec = [], [], [], []
        model.train()
        
        for epoch in range(self.epochs):
            with tqdm(total=len(self.train_set), desc=f'Train', unit='batch') as pbar:
                for i, (feature, label) in enumerate(self.train_set):
                    feature, label = feature.to(self.device), label.to(self.device)
                    optimizer.zero_grad()
                    predict = model(feature)
                    loss = self.criterion(predict, label)
                    # cm, r, p, iou, tn, fp, fn, tp = calc_metrics(predict, label, self.cls_thres)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    # lr_vec.append(optimizer.param_groups[0]["lr"].item())
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()

            train_loss /= len(self.train_set)
            loss_vec.append(train_loss)
            info = f"Epoch: {epoch + 1}\tTraining Loss: {train_loss}\t"
            if (epoch + 1) % self.val_step:
                self.logger.info(info)
            else:
                val_loss, recall, precision, IOU = self.validate(args.cls_thres)
                info += f"\tValidation Loss: {val_loss}\tRecall: {recall}\tPrecision: {precision}\tIOU: {IOU}"
                self.logger.info(info)
                val_vec.append(epoch + 1)
                val_loss_vec.append(val_loss)

                self.save_model(self.weights_path, 'last')
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    self.save_model(self.weights_path, 'best')
            if epoch + 1 == self.epochs:
                r_vec, p_vec, iou_vec = [], [], []
                for c in np.linspace(0, 1, 101):
                    _, r, p, iou = self.validate(c)
                    r_vec.append(r)
                    p_vec.append(p)
                    iou_vec.append(iou)
                
                plot_prcurve(r_vec, p_vec, iou_vec, self.save_path)
            
        self.logger.info("Training Completed.")
        plot_lr_curve(lr_vec, self.save_path)
        plot_loss_curve(loss_vec, val_vec, val_loss_vec, self.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cls_thres', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=int, nargs='+', default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--use_poison', action='store_true')
    parser.add_argument('--data_path', type=str, default='data/SIDD_DNG')
    parser.add_argument('--model_path', type=str, default='UNet')
    args = parser.parse_args()
    pc = PixelCalculate(args)
    pc.train()