import os

import torch
from pytorch_trainer import Module
from torch import nn, optim
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.data.lrw import LRWDataset
from src.models.lrw_model import accuracy
from src.models.nll_sequence_loss import NLLSequenceLoss
from src.models.resnet import ResNetModel


class JoinedExpertModel(Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.left_expert = Expert(resnet_layers=hparams.resnet)
        load_checkpoint(hparams.checkpoint, self.left_expert, strict=False)

        self.center_expert = Expert(resnet_layers=hparams.resnet)
        load_checkpoint(hparams.checkpoint, self.center_expert, strict=False)

        self.right_expert = Expert(resnet_layers=hparams.resnet)
        load_checkpoint(hparams.checkpoint, self.right_expert, strict=False)

        self.joined_backend = JoinedBackend(num_classes=hparams.words)
        load_checkpoint(hparams.checkpoint, self.joined_backend, strict=False)
        self.loss = self.joined_backend.loss

        self.epoch = 0
        self.best_val_acc = 0

    def forward(self, x, yaws):
        left_samples = torch.FloatTensor([]).cuda()
        center_samples = torch.FloatTensor([]).cuda()
        right_samples = torch.FloatTensor([]).cuda()
        for i, yaw in enumerate(yaws):
            if yaw < -20:
                left_samples = torch.cat([left_samples, x[i]])
            elif yaw >= -20 and yaw < 20:
                center_samples = torch.cat([center_samples, x[i]])
            else:
                right_samples = torch.cat([right_samples, x[i]])

        expert_output = []
        if len(left_samples) > 0:
            left_samples = left_samples.unsqueeze(dim=1)
            left = self.left_expert(left_samples)
            expert_output.append(left)
        if len(center_samples) > 0:
            center_samples = center_samples.unsqueeze(dim=1)
            center = self.center_expert(center_samples)
            expert_output.append(center)
        if len(right_samples) > 0:
            right_samples = right_samples.unsqueeze(dim=1)
            right = self.right_expert(right_samples)
            expert_output.append(right)

        output = torch.cat(expert_output)
        output = self.joined_backend(output)
        return output

    def training_step(self, batch):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_step(self, batch):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output = self.forward(frames, yaws)
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

    def train_dataloader(self):
        train_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, seed=self.hparams.seed)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, mode='val', seed=self.hparams.seed)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return val_loader

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
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.resnet(x)
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
