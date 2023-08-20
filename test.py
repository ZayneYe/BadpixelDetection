import argparse
import torch
from tqdm import tqdm
from utils.metrics import calc_metrics, dice_loss
from utils.process import postprocess, generate_pred_dict
from utils.plot import plot_prcurve
from dataset import SamsungDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os


def test(args, thres):
    dataset = args.data_path.split("/")[1][:3]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([151.78], [48.85])])
    mask_transform = transforms.Compose([transforms.ToTensor()])
    test_data = SamsungDataset(args.data_path, cate='test', transform=transform, mask_transform=mask_transform)
    test_set = DataLoader(test_data, batch_size=1, num_workers=args.num_workers, shuffle=False)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path).to(device)
    criterion = torch.nn.BCELoss()
    model.eval()
    test_loss = 0
    pred_dict = {}
    with torch.no_grad():
        with tqdm(total=len(test_set), desc=f'Eval', unit='batch') as pbar:
            for i, (feature, label, file) in enumerate(test_set):
                feature, label = feature.to(device), label.to(device)
                predict = model(feature)
                pred_dict = generate_pred_dict(pred_dict, file, predict, label)
                loss = criterion(predict, label)
                loss += dice_loss(predict, label)
                test_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
            pbar.close()
    test_loss /= len(test_set)
    pred_all, lab_all = postprocess(pred_dict, dataset, args.patch_num)
    recall, precision, iou, dice_score, _ = calc_metrics(pred_all, lab_all, thres)
    return test_loss, recall.item(), precision.item(), iou.item(), dice_score.item()

def lanuch(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    r_vec, p_vec, iou_vec, dice_vec = [], [], [], []
    if not args.test_once:
        for c in np.linspace(0, 1, 31):
            _, r, p, iou, dice = test(args, c)
            # print(r, p, c)
            r_vec.append(r)
            p_vec.append(p)
            iou_vec.append(iou)
            dice_vec.append(dice)
        plot_prcurve(r_vec, p_vec, iou_vec, dice_vec, args.save_path)
    else:
        _, r, p, iou, dice = test(args, 0.5)
        print(f"Recall: {r}, Precision: {p}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_num', type=int, default=4)
    parser.add_argument('--test_once', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default='data/ISP_0.7')
    parser.add_argument('--model_path', type=str, default='runs/train/UNet_ISP_0.7/exp1/weights/best.pt')
    parser.add_argument('--save_path', type=str, default='runs/test/UNet_ISP_0.7_2')
    parser.add_argument('--device', type=int, nargs='+', default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    lanuch(args)