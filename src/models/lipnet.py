import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader


class LipNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
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
        self.pred = nn.Linear(self.rnn_size * 2, vocab_size + 1)

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
