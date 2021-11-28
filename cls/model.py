import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet18, resnet34
import torchvision

from utils import get_tranform, get_split_lenghts, get_model, SamaraMetric


pl.seed_everything(2)



class Resnet(pl.LightningModule):
    def __init__(self, path_to_weights=None, name="patched_resnet", lr=3e-4, size_patch=64, n_classes=20):
        super(Resnet, self).__init__()
        self.model = get_model(path_to_weights, name, n_classes)

        # self.criterion = LabelSmoothingLoss(n_classes, 0.3)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([2.5, 2.5, 1]), 
            label_smoothing=0.2
        )

        self.name = name
        self.lr = lr
        self.size_patch = size_patch
        self.softmax = torch.nn.Softmax()
        self.n_classes = n_classes

        if self.lr is None:
            self.model.eval()
        
        
        # cause logits loss
        self.train_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.train_samara = SamaraMetric()
        # self.train_recall = torchmetrics.Recall()
        # self.train_f1 = torchmetrics.F1()
        # self.train_precision = torchmetrics.Precision()
        # self.train_auc = torchmetrics.AUROC(average="weighted", pos_label=1)

        self.test_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.test_samara = SamaraMetric()
        # self.test_recall = torchmetrics.Recall()
        # self.test_precision = torchmetrics.Precision()
        # self.test_f1 = torchmetrics.F1()
        # self.test_auc = torchmetrics.AUROC(average="weighted", pos_label=1)

        self.val_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.val_samara = SamaraMetric()
        # self.val_recall = torchmetrics.Recall()
        # self.val_f1 = torchmetrics.F1()
        # self.val_precision = torchmetrics.Precision()
        # self.val_auc = torchmetrics.AUROC(average="weighted", pos_label=1)



    def forward(self, x):
        if self.name == "patched_resnet":
            B, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
            Y = H // self.size_patch
            X = W // self.size_patch

            x = x.view(B, C, Y, self.size_patch, X, self.size_patch)
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.contiguous().view(B * Y * X, C, self.size_patch, self.size_patch)
            x = self.model(x)
            x = x.reshape(B, -1, self.n_classes)
            x = x.mean(axis=(1, 2)).unsqueeze(1)
            return self.softmax(x)

        else:
            return self.softmax(self.model(x))

    def loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        probs = self.forward(x)
        loss = self.loss(probs, y)

        self.train_acc(probs, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_samara(probs, y)
        self.log("train_samara", self.val_samara, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        probs = self.forward(x)
        loss = self.loss(probs, y)
        
        self.val_acc(probs, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        
        self.val_samara(probs, y)
        self.log("val_samara", self.val_samara, on_epoch=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        probs = self.forward(x)
        loss = self.loss(probs, y)
        
        self.test_acc(probs, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        self.test_samara(probs, y)
        self.log("test_samara", self.val_samara, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.lr
        )
        return optimizer


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_path, test_path, num_workers):
        super(ImageDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_path = train_path
        self.test_path = test_path

    def setup(self, stage):
        dataset = torchvision.datasets.ImageFolder(self.train_path, get_tranform)
        self.train_dataset = dataset

        # self.train_dataset, self.valid_dataset = random_split(dataset, get_split_lenghts(dataset))
        self.valid_dataset = torchvision.datasets.ImageFolder(self.test_path, get_tranform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


class MyCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_torch_weights(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        monitor_candidates = self._monitor_candidates(trainer, epoch=epoch, step=global_step)
        current = monitor_candidates.get(self.monitor)

        if self.check_monitor_top_k(trainer, current):
            torch.save(pl_module.model.state_dict(), os.path.join(self.dirpath, "model.pth"))

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if (
            self._should_skip_saving_checkpoint(trainer)
            or self._save_on_train_epoch_end
            or self._every_n_epochs < 1
            or (trainer.current_epoch + 1) % self._every_n_epochs != 0
        ):
            return

        self.save_torch_weights(trainer, pl_module)
        self.save_checkpoint(trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    opt = OmegaConf.load(args.c) # config

    name = args.name + datetime.strftime(datetime.now(), "_%h%d_%H%M%S")
    opt.experiment_dir = str(Path(opt.checkpoints_dir, "experiments", name))
    Path(opt.experiment_dir).mkdir(parents=True, exist_ok=True)

    # setup_logger(opt.experiment_dir, fname=f"{name}_main.log")
    wandb_logger = WandbLogger(
        project=opt.project_name, entity=opt.entity, save_dir=opt.experiment_dir, config=vars(opt), name=name, dir="/tmp",
    )

    data_module = ImageDataModule(**opt.data)
    model = Resnet(**opt.model)

    checkpoint_callback = MyCheckpoint(
        monitor=opt.metric,
        dirpath=opt.experiment_dir,
        filename="{epoch}-{val_auc:.2f}-{val_f1:.2f}",
        save_top_k=1,
        verbose=True,
        mode="max",
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        gpus=[opt.device],
        logger=wandb_logger,
        max_epochs=opt.max_epoch,
        callbacks=[
            EarlyStopping(opt.metric, verbose=True, mode="max", patience=opt.early_stoping_patience),
            checkpoint_callback,
        ],
    )

    trainer.fit(model, data_module)

    wo_weights = {key: value for key, value in opt.model.items() if key != "path_to_weights"}
    model = Resnet(**wo_weights, path_to_weights=os.path.join(opt.experiment_dir, "model.pth"))
    model.eval()

    trainer.test(model, datamodule=data_module)