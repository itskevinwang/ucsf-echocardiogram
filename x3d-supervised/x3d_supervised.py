import argparse
import json
import logging
import os
import sys
from typing import Dict

import albumentations as albu
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning
import pytorchvideo
import pytorchvideo.data
from torchmetrics import MetricCollection, CosineSimilarity, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, SymmetricMeanAbsolutePercentageError

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from albumentations.pytorch import ToTensorV2

class X3D(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=pretrained)
        self.fc = nn.Linear(400, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.__configure_metrics()
        if self.args.arch in ["x3d_xs", "x3d_s", "x3d_m", "x3d_l"]:
            self.model = X3D(model_name=self.args.arch)
        else:
            raise ValueError(f"Model {self.args.arch} not supported.")

    def __configure_metrics(self):
        self.train_metrics = [
            MetricCollection(
                [
                    MeanAbsoluteError(),
                    MeanSquaredError(),
                ],
                prefix='train/'
            )
        ]
        self.val_metrics = [
            MetricCollection(
                [
                    CosineSimilarity(),
                    MeanAbsoluteError(),
                    MeanSquaredError(),
                    PearsonCorrCoef(),
                    SymmetricMeanAbsolutePercentageError(),
                ],
                prefix='val/'
            )
        ]

    def forward(self, x):
        logits = self.model(x)
        probs = F.log_softmax(logits, dim=1)
        return probs

    def training_step(self, batch, batch_idx):
        x = batch["data"].float()
        labels = batch["label"]

        logits = self.model(x)
        # loss subject to change
        batch_loss = F.mse_loss(logits, labels)

        self.log("train/loss", batch_loss)
        return {'loss': batch_loss, 'preds': logits.detach(), 'target': labels.detach()}

    def training_step_end(self, outputs):
        for train_metrics in self.train_metrics:
            train_metrics = train_metrics.to(self.device)
            metrics = train_metrics(outputs['preds'], outputs['target'])
            self.log_dict(metrics, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x = batch["data"].float()
        labels = batch["label"]

        logits = self.model(x)
        # loss subject to change
        batch_loss = F.mse_loss(logits, labels)
    
        self.log("val/loss", batch_loss)
        return {'loss': batch_loss, 'preds': logits.detach(), 'target': labels.detach()}

    def validation_step_end(self, outputs):
        for val_metrics in self.val_metrics:
            val_metrics = val_metrics.to(self.device)
            metrics = val_metrics(outputs['preds'], outputs['target'])
            self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]

def sample_vid_frames(video_path, vid_NumberOfFrames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    i = 0
    sample_rate = int(vid_NumberOfFrames/16)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % sample_rate == 0 and len(frames) < 16: 
            frames.append(frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames)

class EchoDataset(Dataset):
    def __init__(self, df, VID_PATH, mode):
        self.df = df
        self.mode = mode
        self.VID_PATH = VID_PATH

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        vid_FileName = row.FileName 
        vid_FPS = row.FPS
        vid_NumberOfFrames = row.NumberOfFrames
        vid_EF = np.array([row.EF])
        vid_duration = vid_NumberOfFrames/vid_FPS
        VID_PATH = self.VID_PATH
        video_path = VID_PATH + vid_FileName + '.mp4'
        start_sec = 0
        end_sec = vid_duration
        video_data = sample_vid_frames(video_path, vid_NumberOfFrames)

        # Note to self: do not use ToTensorV2(). It messes up the shape :(
        self.train_transform = albu.Compose([
              # need to add some augmentation here
              # ToTensorV2(),
          ])

        self.valid_transform = albu.Compose([
              # ToTensorV2(),
          ])
        output_frames = []
        if len(video_data) >= 16:
          for i in range(16):
            # BE SURE TO CHECK IF THE TRANSPOSED SHAPE MAKES SENSE
            image = video_data[i]
            image = image.transpose((2,1,0))
            if self.mode == 'train':
              output_frames.append(self.train_transform(image=image)['image'])
            else: 
              output_frames.append(self.valid_transform(image=image)['image'])
        else:
          output_frames = [f for f in video_data]
	
        output_frames = np.stack(output_frames)
        output_frames = output_frames.transpose((1,0,2,3))
        output_frames = torch.from_numpy(output_frames)

        output = {'data': output_frames, 'label': torch.from_numpy(vid_EF)}
        return output

class EchoDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, args):
        self.args = args
        super().__init__()

    def train_dataloader(self):
        df = pd.read_csv(self.args.DF_PATH)
        df_train = df[df.Split == 'TRAIN']

        self.train_dataset = EchoDataset(df_train, self.args.VID_PATH, 'train')

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):
        df = pd.read_csv(self.args.DF_PATH)
        df_val = df[df.Split == 'VAL']

        self.val_dataset = EchoDataset(df_val, self.args.VID_PATH, 'val')
        print(len(self.val_dataset))

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

def main():
    setup_logger()
    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="x3d_s",
        choices=["x3d_xs, x3d_s, x3d_m, x3d_l"],
        type=str
    )

    # Data parameters
    parser.add_argument("--DF_PATH", default="/echodl/data/metadata/FileList.csv", type=str)
    parser.add_argument("--VID_PATH", default="/echodl/data/echo_resized_224/", type=str)

    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)

    # Trainer parameters
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=5,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
        gpus=1
    )

    args = parser.parse_args(args=[])
    train(args)

def train(args):
    wandb_logger = WandbLogger(entity="ucsfechocardiogram", project="x3dsupervised")
    trainer = pytorch_lightning.Trainer.from_argparse_args(args, logger=wandb_logger)
    classification_module = VideoClassificationLightningModule(args)
    data_module = EchoDataModule(args)
    trainer.fit(classification_module, data_module)

def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)

if __name__ == "__main__":
    main()
