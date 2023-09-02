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
from utils.plot import plot_loss_curve, plot_prcurve, plot_lr_curve, plot_iou_curve
from utils.metrics import calc_metrics, dice_loss, calc_metrics_one
from utils.process import postprocess, generate_pred_dict
from torch.optim.lr_scheduler import StepLR
import numpy as np


class PixelCalculate():
    def __init__(self, args):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([151.78], [48.85])])
        mask_transform = transforms.Compose([transforms.ToTensor()])
        train_data = SamsungDataset(args.data_path, cate='train', transform=transform, mask_transform=mask_transform, patch_num=args.patch_num)
        val_data = SamsungDataset(args.data_path, cate='val', transform=transform, mask_transform=mask_transform, patch_num=args.patch_num)
        
        self.patch_num = args.patch_num
        self.train_set = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        self.val_set = DataLoader(val_data, batch_size=1, num_workers=args.num_workers, shuffle=False)
        self.dataset = args.data_path.split("/")[1][:3]
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
        pred_dict = {}
        with torch.no_grad():
            with tqdm(total=len(self.val_set), desc=f'Eval', unit='batch') as pbar:
                for i, (feature, label, file) in enumerate(self.val_set):
                    feature, label = feature.to(self.device), label.to(self.device)
                    predict = self.model(feature)
                    pred_dict = generate_pred_dict(pred_dict, file, predict, label)
                    loss = self.criterion(predict, label)
                    loss += dice_loss(predict, label)
                    val_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
        val_loss /= len(self.val_set)
        pred_all, lab_all = postprocess(pred_dict, self.dataset, self.patch_num)
        if self.dataset == "ISP":
            recall, precision, iou, dice_score, _ = calc_metrics(pred_all, lab_all, thres)
        else:
            recall, precision, iou, dice_score = calc_metrics_one(pred_all, lab_all, thres)
        
        return val_loss, recall.item(), precision.item(), iou.item(), dice_score.item()


    def train(self):
        self.logger.info("Start Training...")
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
        
        train_loss, val_loss, max_iou = 0, 0, 0
        loss_vec, val_loss_vec, val_vec, lr_vec, iou_score_vec, dice_score_vec = [], [], [], [], [], []
        model.train()
        
        for epoch in range(self.epochs):
            with tqdm(total=len(self.train_set), desc=f'Train', unit='batch') as pbar:
                for i, (feature, label, _) in enumerate(self.train_set):
                    feature, label = feature.to(self.device), label.to(self.device)
                    
                    optimizer.zero_grad()
                    predict = model(feature)
                    loss = self.criterion(predict, label)
                    loss += dice_loss(predict, label)
                    # cm, r, p, iou, tn, fp, fn, tp = calc_metrics(predict, label, self.cls_thres)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    lr_vec.append(optimizer.param_groups[0]["lr"])
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                pbar.close()
            scheduler.step()
            train_loss /= len(self.train_set)
            loss_vec.append(train_loss)
            info = f"Epoch: {epoch + 1}\tTraining Loss: {train_loss:.4f}\t"
            if (epoch + 1) % self.val_step:
                self.logger.info(info)
            else:
                val_loss, recall, precision, iou_score, dice_score = self.validate(args.cls_thres)
                info += f"Validation Loss: {val_loss:.4f}\tRecall: {recall:.4f}\tPrecision: {precision:.4f}\tIOU: {iou_score:.4f}\tDice score: {dice_score:.4f}"
                self.logger.info(info)
                val_vec.append(epoch + 1)
                val_loss_vec.append(val_loss)
                iou_score_vec.append(iou_score)
                dice_score_vec.append(dice_score)
                self.save_model(self.weights_path, 'last')
                if iou_score > max_iou:
                    max_iou = iou_score
                    self.save_model(self.weights_path, 'best')
            if epoch + 1 == self.epochs:
                r_vec, p_vec, iou_vec, dice_vec = [], [], [], []
                for c in np.linspace(0, 1, 31):
                    _, r, p, iou, dice = self.validate(c)
                    # print(r, p, c)
                    r_vec.append(r)
                    p_vec.append(p)
                    iou_vec.append(iou)
                    dice_vec.append(dice)
                
                plot_prcurve(r_vec, p_vec, iou_vec, dice_vec, self.save_path)
            
        self.logger.info("Training Completed.")
        plot_lr_curve(lr_vec, self.save_path)
        plot_loss_curve(loss_vec, val_vec, val_loss_vec, self.save_path)
        plot_iou_curve(iou_score_vec, dice_score_vec, val_vec, self.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cls_thres', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=int, nargs='+', default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--patch_num', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='/data1/Invertible_ISP/Invertible_ISP_0.7')
    parser.add_argument('--model_path', type=str, default='Invertible_ISP_0.7_4')
    args = parser.parse_args()
    pc = PixelCalculate(args)
    pc.train()