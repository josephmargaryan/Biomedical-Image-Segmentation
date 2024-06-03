from dataset import val_loader 
import numpy as np 
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unet import UNET
from dataset import device

class SegmentationMetrics:
    def __init__(self, model, val_loader):
        self.model = model
        self.val_loader = val_loader
    
    def IoU(self):
        iou_scores = []
        for x, y in tqdm(self.val_loader):
            x, y = x.to(device), y.to(device)
            self.model.eval()
            with torch.no_grad():
                out = (F.sigmoid(self.model(x)) > 0.5).float()
                preds = out.cpu().numpy()
                gt = y.cpu().numpy()
                
                tp = np.sum((preds == 1) & (gt == 1))
                fp = np.sum((preds == 1) & (gt == 0))
                fn = np.sum((preds == 0) & (gt == 1))
                
                iou = tp / (tp + fp + fn + 1e-10)
                iou_scores.append(iou)
        
        mean_iou = np.mean(iou_scores)
        return f'Mean Binary IoU: {mean_iou}'
    
    def Dice(self):
        dice_scores = []
        for x, y in tqdm(self.val_loader):
            x, y = x.to(device), y.to(device)
            self.model.eval()
            with torch.no_grad():
                out = (F.sigmoid(self.model(x)) > 0.5).float()
                preds = out.cpu().numpy()
                gt = y.cpu().numpy()
                
                tp = np.sum((preds == 1) & (gt == 1))
                fp = np.sum((preds == 1) & (gt == 0))
                fn = np.sum((preds == 0) & (gt == 1))
                
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-10)
                dice_scores.append(dice)
        
        mean_dice = np.mean(dice_scores)
        return f'Mean Binary Dice: {mean_dice}'
            