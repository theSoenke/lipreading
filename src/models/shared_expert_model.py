import os

import matplotlib.pyplot as plt
import torch
from pytorch_trainer import Module, data_loader
from torch import nn, optim
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.data.lrw import LRWDataset
from src.models.attention import Attention
from src.models.lrw_model import accuracy
from src.models.nll_sequence_loss import NLLSequenceLoss
from src.models.resnet import BasicBlock, ResNet, ResNetModel


class SharedExpertModel(Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.center_expert = Expert(resnet_layers=hparams.resnet)
        self.block1 = Block()
        self.joined_backend = JoinedBackend(num_classes=hparams.words)

        self.loss = self.joined_backend.loss
        self.attention = Attention(attention_dim=40, num_experts=3)

        self.logger = None
        self.epoch = 0
        self.best_val_acc = 0

    def forward(self, x, yaws):
        output = self.center_expert(x)
        output = self.block1(output)
        output = output.view(x.shape[0], -1, 256)
        output = self.joined_backend(output)
        return output, None

    def training_step(self, batch):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output, _ = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_step(self, batch):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output, attn = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)

        return {
            'val_loss': loss,
            'val_acc': acc,
            'yaws': yaws,
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        if self.best_val_acc < avg_acc:
            self.best_val_acc = avg_acc
        logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'best_val_acc': self.best_val_acc,
        }
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'log': logs,
        }

    def test_step(self, batch):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output, _ = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        return {
            'test_loss': loss,
            'test_acc': acc,
        }

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {
            'test_loss': avg_loss,
            'test_acc': avg_acc,
            'log': logs,
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @data_loader
    def train_dataloader(self):
        train_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, seed=self.hparams.seed)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True)
        return train_loader

    @data_loader
    def val_dataloader(self):
        val_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, mode='val', seed=self.hparams.seed)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return val_loader

    @data_loader
    def test_dataloader(self):
        test_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, mode='test', seed=self.hparams.seed)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return test_loader


class Expert(nn.Module):
    def __init__(self, resnet_layers=18):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = ResNetModel(layers=resnet_layers, pretrained=False)


    def forward(self, x):
        x = self.frontend(x)
        x = self.resnet(x)

        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )

        self.block1 = BasicBlock(256, 512, stride=2, downsample=downsample)
        self.block2 = BasicBlock(512, 512)

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)

        return x


class JoinedBackend(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256 * 2, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss = NLLSequenceLoss()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
