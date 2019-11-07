import os

import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_trainer import Module
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

import wandb
from src.data.ctc_utils import ctc_collate
from src.data.lrs2 import LRS2Dataset
from src.decoder.greedy import GreedyDecoder
from src.models.resnet import ResNetModel


class LRS2Model(Module):
    def __init__(self, hparams, in_channels=1, augmentations=False, pretrain=False):
        super().__init__()
        self.hparams = hparams
        self.in_channels = in_channels
        self.augmentations = augmentations
        self.pretrain = pretrain
        self.max_timesteps = 155
        self.pretrain_words = 0

        characters = self.train_dataloader().dataset.characters
        self.decoder = GreedyDecoder(characters)
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
        self.fc = nn.Linear(512, len(characters))
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss = nn.CTCLoss(reduction='none', zero_infinity=True)

        self.best_val_wer = 1.0
        self.epoch = 0

    def forward(self, x, lengths):
        # x = x.narrow(2, 0, max(lengths))
        x = self.frontend(x)
        x = self.resnet(x)
        x = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_num):
        frames, y, lengths, y_lengths, idx = batch
        output = self.forward(frames, lengths)
        output = output.transpose(0, 1)
        loss_all = self.loss(output, y, lengths, y_lengths)
        loss = loss_all.mean()

        weight = torch.ones_like(loss_all)
        dlogits = torch.autograd.grad(loss_all, output, grad_outputs=weight)[0]
        output.backward(dlogits)

        logs = {'train_loss': loss}
        if batch_num % 50 == 0:
            predicted, gt, samples = self.decoder.predict(frames.size(0), output, y, lengths, y_lengths, n_show=3)
            wer = self.decoder.wer_batch(predicted, gt)
            cer = self.decoder.cer_batch(predicted, gt)
            logs = {'train_loss': loss, 'train_cer': cer, 'train_wer': wer}
            print(samples)

        return {'log': logs}

    def validation_step(self, batch, batch_num):
        if self.pretrain:
            return {}

        frames, y, lengths, y_lengths, idx = batch

        output = self.forward(frames, lengths)
        output = output.transpose(0, 1)
        loss_all = self.loss(output, y, lengths, y_lengths)
        loss = loss_all.mean()

        predicted, gt, samples = self.decoder.predict(frames.size(0), output, y, lengths, y_lengths, n_show=3)
        return {
            'val_loss': loss,
            'predictions': predicted,
            'ground_truth': gt,
            'samples': samples,
        }

    def validation_end(self, outputs):
        if self.pretrain:
            print("Skip validation for pretraining")
            return {}

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        predictions = np.concatenate([x['predictions'] for x in outputs])
        ground_truth = np.concatenate([x['ground_truth'] for x in outputs])
        samples = np.concatenate([x['samples'] for x in outputs])
        wer = self.decoder.wer_batch(predictions, ground_truth)
        cer = self.decoder.cer_batch(predictions, ground_truth)

        print(samples)

        # self.logger.log_metrics({"samples": wandb.Table(data=samples, columns=["Sentence", "Predicted"])})

        if self.best_val_wer > wer:
            self.best_val_wer = wer
        logs = {
            'val_loss': avg_loss,
            'val_cer': cer,
            'val_wer': wer,
            'best_val_wer': self.best_val_wer
        }

        self.epoch += 1
        return {
            'val_loss': avg_loss,
            'val_wer': wer,
            'val_cer': cer,
            'log': logs,
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        if self.pretrain:
            mode = "pretrain"
        else:
            mode = "train"
        train_data = LRS2Dataset(
            path=self.hparams.data,
            in_channels=self.in_channels,
            augmentations=self.augmentations,
            mode=mode,
            max_timesteps=self.max_timesteps,
            pretrain_words=self.pretrain_words,
        )
        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
            pin_memory=True,
            collate_fn=ctc_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_data = LRS2Dataset(
            path=self.hparams.data,
            in_channels=self.in_channels,
            mode='val',
        )
        val_loader = DataLoader(
            val_data, shuffle=False,
            batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers,
            collate_fn=ctc_collate,
        )
        return val_loader

    def test_dataloader(self):
        test_data = LRS2Dataset(
            path=self.hparams.data,
            in_channels=self.in_channels,
            mode='test',
        )
        test_loader = DataLoader(
            test_data, shuffle=False,
            batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers,
            collate_fn=ctc_collate,
        )
        return test_loader


def accuracy(output, labels):
    sums = torch.sum(output, dim=1)
    _, predicted = sums.max(dim=1)
    correct = (predicted == labels.squeeze(dim=1)).sum().type(torch.FloatTensor)
    return correct / output.shape[0]
