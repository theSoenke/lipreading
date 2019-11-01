import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_trainer import Module, data_loader
from torch import nn as nn
from torch import optim
from torch.nn import functional as F
from torch.nn import init
from torch.utils.data import DataLoader

from src.data.ctc_utils import ctc_collate
from src.data.grid import GRIDDataset
from src.models.ctc_decoder import Decoder


class LipNet(Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.vocab = GRIDDataset(path=self.hparams.data).vocab
        self.vocab_size = len(self.vocab)
        self.decoder = Decoder(self.vocab)
        self.loss = nn.CTCLoss(reduction='none', zero_infinity=True)
        self.dropout = 0.5
        self.rnn_size = 256
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(self.dropout)
        )
        self.gru1 = nn.GRU(32 * 3 * 6, self.rnn_size, 1, bidirectional=True)
        self.drp1 = nn.Dropout(self.dropout)
        self.gru2 = nn.GRU(self.rnn_size * 2, self.rnn_size, 1, bidirectional=True)
        self.drp2 = nn.Dropout(self.dropout)
        self.pred = nn.Linear(self.rnn_size * 2, self.vocab_size + 1)

        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        init.kaiming_normal_(self.pred.weight, nonlinearity='sigmoid')
        init.constant_(self.pred.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (32 * 3 * 6 + self.rnn_size))
            for i in range(0, self.rnn_size * 3, self.rnn_size):
                init.uniform_(m.weight_ih_l0[i: i + self.rnn_size], -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + self.rnn_size])
                init.constant_(m.bias_ih_l0[i: i + self.rnn_size], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + self.rnn_size], -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + self.rnn_size])
                init.constant_(m.bias_ih_l0_reverse[i: i + self.rnn_size], 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.gru1(x)
        x = self.drp1(x)
        x, _ = self.gru2(x)
        x = self.drp2(x)
        x = self.pred(x)

        return x

    def training_step(self, batch, batch_nb):
        x, y, lengths, y_lengths, idx = batch
        logits = self.forward(x)
        loss_all = self.loss(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
        loss = loss_all.mean()

        weight = torch.ones_like(loss_all)
        dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]
        logits.backward(dlogits)

        logs = {'train_loss': loss}
        return {'log': logs}

    def validation_step(self, batch, batch_nb):
        x, y, lengths, y_lengths, idx = batch
        logits = self.forward(x)
        loss_all = self.loss(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
        loss = loss_all.mean()

        decoded, gt, _ = self.decoder.predict(x.size(0), logits, y, lengths, y_lengths, n_show=5, mode='greedy')
        return {'val_loss': loss, 'pred': decoded, 'gt': gt}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        predictions = np.concatenate([x['predictions'] for x in outputs])
        ground_truth = np.concatenate([x['ground_truth'] for x in outputs])

        wer = self.decoder.wer_batch(predictions, ground_truth)
        cer = self.decoder.cer_batch(predictions, ground_truth)

        logs = {'val_loss': avg_loss, 'val_wer': wer, 'val_cer': cer}
        return {
            'val_loss': avg_loss,
            'log': logs,
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    @data_loader
    def train_dataloader(self):
        train_data = GRIDDataset(path=self.hparams.data, augmentation=True)
        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            collate_fn=ctc_collate,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )
        return train_loader

    @data_loader
    def val_dataloader(self):
        val_data = GRIDDataset(path=self.hparams.data, mode='val')
        val_loader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=self.hparams.batch_size * 2,
            collate_fn=ctc_collate,
            num_workers=self.hparams.workers,
        )
        return val_loader

    @data_loader
    def test_dataloader(self):
        val_data = GRIDDataset(path=self.hparams.data, mode='test')
        val_loader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=self.hparams.batch_size * 2,
            collate_fn=ctc_collate,
            num_workers=self.hparams.workers,
        )
        return val_loader
