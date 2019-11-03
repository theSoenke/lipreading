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
from src.models.resnet import ResNetModel


class ExpertEarlyAttnModel(Module):
    def __init__(self, hparams, ckpt_left, ckpt_center, ckpt_right):
        super().__init__()
        self.hparams = hparams
        self.logger = None

        self.left_expert = Expert(hparams.words, in_channels=1, resnet_layers=hparams.resnet)
        load_checkpoint(ckpt_left, self.left_expert, strict=False)

        self.center_expert = Expert(hparams.words, in_channels=1, resnet_layers=hparams.resnet)
        load_checkpoint(ckpt_center, self.center_expert, strict=False)

        self.right_expert = Expert(hparams.words, in_channels=1, resnet_layers=hparams.resnet)
        load_checkpoint(ckpt_right, self.right_expert, strict=False)

        self.joined_backend = JoinedBackend(num_classes=hparams.words)
        load_checkpoint(ckpt_center, self.joined_backend, strict=False)
        self.joined_backend.freeze()

        self.loss = NLLSequenceLoss()
        self.attention = Attention(attention_dim=40, num_experts=3)
        self.softmax = nn.LogSoftmax(dim=2)

        self.epoch = 0
        self.best_val_acc = 0

    def forward(self, x, yaws):
        left = self.left_expert(x)
        center = self.center_expert(x)
        right = self.right_expert(x)
        context = self.attention(yaws)
        attn = context.split(split_size=1, dim=1)

        left_flat = left.view(x.size(0), -1) * attn[0]
        center_flat = center.view(x.size(0), -1) * attn[1]
        right_flat = right.view(x.size(0), -1) * attn[2]
        output = (left_flat + center_flat + right_flat).view(x.size(0), 29, 256)
        output = self.joined_backend(output)

        return output, attn

    def training_step(self, batch, batch_num):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output, attn = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_step(self, batch, batch_num):
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
            'left_attn': attn[0],
            'center_attn': attn[1],
            'right_attn': attn[2],
        }

    def validation_end(self, outputs):
        self.visualize_attention(outputs)

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

    def test_step(self, batch, batch_num):
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

    def visualize_attention(self, outputs):
        yaws = torch.cat([x['yaws'] for x in outputs]).squeeze(dim=1).cpu().numpy()
        left_attn = torch.cat([x['left_attn'] for x in outputs]).squeeze(dim=1).cpu().numpy()
        center_attn = torch.cat([x['center_attn'] for x in outputs]).squeeze(dim=1).cpu().numpy()
        right_attn = torch.cat([x['right_attn'] for x in outputs]).squeeze(dim=1).cpu().numpy()

        size = 40
        plt.scatter(yaws, left_attn, s=size, label='left expert')
        plt.scatter(yaws, center_attn, s=size, label='center expert')
        plt.scatter(yaws, right_attn, s=size, label='right expert')
        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.xlabel('Degree')
        plt.ylabel('Attention')

        directory = "data/viz/lrw"
        os.makedirs(directory, exist_ok=True)
        path = f"{directory}/attention_seed_{self.hparams.seed}_epoch_{self.epoch}.png"
        plt.savefig(path)
        self.logger.save_file(path)
        plt.clf()
        self.epoch += 1

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
    def __init__(self, num_classes, in_channels=1, resnet_layers=13):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = ResNetModel(layers=resnet_layers, pretrained=False)
        self.freeze()

    def forward(self, x):
        x = self.frontend(x)
        x = self.resnet(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


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

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
