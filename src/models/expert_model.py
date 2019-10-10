import os

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.data.lrw import LRWDataset
from src.models.attention import Attention
from src.models.lrw_model import LRWModel
from src.models.nll_sequence_loss import NLLSequenceLoss


class ExpertModel(pl.LightningModule):
    def __init__(self, hparams, ckpt_left, ckpt_center, ckpt_right):
        super().__init__()
        self.hparams = hparams

        self.left_expert = LRWModel(hparams, query=(-90, -20))
        load_checkpoint(ckpt_left, self.left_expert)
        self.left_expert.freeze()

        self.center_expert = LRWModel(hparams, query=(-20, 20))
        load_checkpoint(ckpt_center, self.center_expert)
        self.center_expert.freeze()

        self.right_expert = LRWModel(hparams, query=(20, 90))
        load_checkpoint(ckpt_right, self.right_expert)
        self.right_expert.freeze()

        self.loss = self.center_expert.loss
        self.attention = Attention(attention_dim=40, num_experts=3)

    def forward(self, x, yaws):
        left = self.left_expert(x)
        center = self.center_expert(x)
        right = self.right_expert(x)
        context = self.attention(yaws)
        expert_attn = context.split(split_size=1, dim=1)

        return left, center, right, expert_attn

    def training_step(self, batch, batch_nb):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        left, center, right, attn = self.forward(frames, yaws)
        left_flat = left.view(frames.size(0), -1) * attn[0]
        center_flat = center.view(frames.size(0), -1) * attn[1]
        right_flat = right.view(frames.size(0), -1) * attn[2]
        output = (left_flat + center_flat + right_flat).view(frames.size(0), 29, 10)

        loss = self.loss(output, labels.squeeze(1))
        acc = self.accuracy(output, labels)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_step(self, batch, batch_nb):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        left, center, right, attn = self.forward(frames, yaws)
        left_flat = left.view(frames.size(0), -1) * attn[0]
        center_flat = center.view(frames.size(0), -1) * attn[1]
        right_flat = right.view(frames.size(0), -1) * attn[2]
        output = (left_flat + center_flat + right_flat).view(frames.size(0), 29, 10)

        loss = self.loss(output, labels.squeeze(1))
        acc = self.accuracy(output, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'log': logs,
        }

    def configure_optimizers(self):
        return optim.Adam(self.attention.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @pl.data_loader
    def train_dataloader(self):
        train_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, seed=self.hparams.seed)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True)
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        val_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, mode='val', seed=self.hparams.seed)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return val_loader

    @pl.data_loader
    def test_dataloader(self):
        train_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, mode='train', seed=self.hparams.seed)
        train_loader = DataLoader(train_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return train_loader

    def accuracy(self, output, labels):
        sums = torch.sum(output, dim=1)
        _, predicted = sums.max(dim=1)
        correct = (predicted == labels.squeeze(dim=1)).sum().type(torch.FloatTensor)
        return correct / output.shape[0]
