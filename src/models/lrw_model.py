import os

import torch
import torchvision.transforms as transforms
from pytorch_trainer import Module, data_loader
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.lrw import LRWDataset
from src.models.nll_sequence_loss import NLLSequenceLoss
from src.models.resnet import ResNetModel


class LRWModel(Module):
    def __init__(self, hparams, in_channels=1, augmentations=False, query=None):
        super().__init__()
        self.hparams = hparams
        self.in_channels = in_channels
        self.augmentations = augmentations
        self.query = query

        self.best_val_acc = 0

        self.frontend = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = ResNetModel(layers=hparams.resnet, pretrained=hparams.pretrained)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256 * 2, hparams.words)
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss = NLLSequenceLoss()

    def forward(self, x):
        x = self.frontend(x)
        x = self.resnet(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch):
        frames = batch['frames']
        labels = batch['label']
        output = self.forward(frames)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_step(self, batch):
        frames = batch['frames']
        labels = batch['label']
        output = self.forward(frames)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        if self.best_val_acc < avg_acc:
            self.best_val_acc = avg_acc
        logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'best_val_acc': self.best_val_acc
        }
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'log': logs,
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @data_loader
    def train_dataloader(self):
        train_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            augmentations=self.augmentations,
            query=self.query,
            seed=self.hparams.seed
        )
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True)
        return train_loader

    @data_loader
    def val_dataloader(self):
        val_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='val',
            query=self.query,
            seed=self.hparams.seed
        )
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return val_loader

    @data_loader
    def test_dataloader(self):
        test_data = LRWDataset(
            path=self.hparams.data,
            num_words=self.hparams.words,
            in_channels=self.in_channels,
            mode='test',
            query=self.query,
            seed=self.hparams.seed
        )
        test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return test_loader


def accuracy(output, labels):
    sums = torch.sum(output, dim=1)
    _, predicted = sums.max(dim=1)
    correct = (predicted == labels.squeeze(dim=1)).sum().type(torch.FloatTensor)
    return correct / output.shape[0]
