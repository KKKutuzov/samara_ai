from typing import List

import numpy as np
import pandas as pd
from torchvision import transforms
import albumentations as A
from torchvision.models import resnet18, resnet34, efficientnet_b6 
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from torchmetrics import Metric


def get_split_lenghts(dataset):
    train_size = int(np.round(0.8 * len(dataset)))
    val_size = int(len(dataset) - train_size)

    return train_size, val_size


def get_tranform(x):
    A_transforms = A.Compose([
        A.HorizontalFlip(),
        A.LongestMaxSize(max_size=600),
        A.PadIfNeeded(600, 600, border_mode=cv2.BORDER_CONSTANT),
        A.Rotate(30),
        A.RandomBrightnessContrast(p=0.2),
        A.Sharpen(alpha=(0.4, 0.5), p=0.2),
        A.MotionBlur(p=0.2, blur_limit=5),
        A.RandomToneCurve(p=0.2),
        A.ColorJitter(p=0.2),
        A.GaussNoise(p=0.2),
        A.GridDistortion(p=0.1),
        A.ISONoise(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Cutout(),
        A.FancyPCA(),
        A.RandomCrop(512, 512),
        A.RandomRain(p=0.1),
    ])

    to_night = A.Compose([
        A.HueSaturationValue(hue_shift_limit=180, p=1),
        A.ToGray(p=1)
    ])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    x = np.array(x, np.uint8)
    alb_x = A_transforms(image=x)['image']

    if np.random.rand() < 0.3:
        alb_x = to_night(image=alb_x)['image']

    torch_x = to_tensor(alb_x)
    return torch_x


def get_model(path_to_weights=None, name="resnet", n_classes=1):
    pretrained = True if path_to_weights is None else False

    if name in ("resnet", "patched_resnet"):
        model = resnet34(pretrained=pretrained)
        model.fc = torch.nn.Linear(512, n_classes)
        
    elif name == "efficient": 
        model = efficientnet_b6(pretrained=pretrained)

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), 
            nn.Linear(in_features=2304, out_features=87, bias=True)
        )
        
        if path_to_weights is not None:
            model.load_state_dict(torch.load(path_to_weights))
            print('Model was loaded!')

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), 
            nn.Linear(in_features=2304, out_features=n_classes, bias=True)
        )
    else:
        NotImplementedError

    params = list(model.parameters())
    for param in params[:int(len(params) * 0.95)]:
        param.requires_grad = False

    return model


class SamaraMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states  
        for pred, tar in zip(preds.argmax(-1), target):
            if pred == tar and pred in (0, 1):
                self.correct += 2
            elif pred == tar and pred == 2:
                self.correct += 0
            elif pred != tar and pred in (0, 1) and tar in (0, 1):
                self.correct += 1
            elif pred in (0, 1) and tar == 2:
                self.correct -= 1
            else:
                self.correct -= 2

        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total