import os
import random
import re

import editdistance
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_trainer import Module
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader

import wandb
from src.data.charset import get_charSet, init_charSet
from src.data.lrs_wls import LRS2Dataset


class WLSNet(Module):
    def __init__(self, hparams, in_channels=1, pretrain=False):
        super().__init__()
        self.hparams = hparams
        self.in_channels = in_channels
        self.pretrain = pretrain
        self.max_timesteps = 155
        self.max_text_len = 100
        self.teacher_forcing_ratio = hparams.teacher_forcing

        init_charSet("en")
        self.watch = Watch(3, 512, 512)
        self.spell = Spell(3, 512, get_charSet().get_total_num())
        self.device = torch.device("cuda:0")
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_cer = 1.0

    def forward(self, x, lengths, target_tensor):
        watch_outputs, watch_state = self.watch(x, lengths)
        decoder_input = torch.tensor([[get_charSet().get_index_of('<sos>')]]).repeat(watch_outputs.size(0), 1).to(self.device)
        spell_hidden = watch_state
        cell_state = torch.zeros_like(spell_hidden).to(self.device)
        context = torch.zeros(watch_outputs.size(0), 1, spell_hidden.size(2)).to(self.device)

        loss = 0
        results = []
        target_length = target_tensor.size(1)
        for di in range(target_length):
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            decoder_output, spell_hidden, cell_state, context = self.spell(decoder_input, spell_hidden, cell_state, watch_outputs, context)
            _, topi = decoder_output.topk(1, dim=2)
            if use_teacher_forcing:
                decoder_input = target_tensor[:, di].long().unsqueeze(dim=1)
            else:
                decoder_input = topi.squeeze(dim=1).detach()

            loss += self.criterion(decoder_output.squeeze(1), target_tensor[:, di].long())
            results.append(topi.cpu().squeeze(1))

        results = torch.cat(results, dim=1)
        return results, loss

    def decode(self, results, target_tensor, batch_num, log_interval=1, log=False):
        cer = 0
        target_length = results.size(1)
        batch_size = results.size(0)
        for batch in range(batch_size):
            output = ''
            label = ''
            for index in range(target_length):
                output += get_charSet().get_char_of(int(results[batch, index]))
                label += get_charSet().get_char_of(int(target_tensor[batch, index]))
            label = label.replace('<pad>', ' ').replace('<eos>', '@')
            label = label[:label.find("@")]
            output = output.replace('<eos>', '@').replace('<pad>', ' ').replace('<sos>', '&')
            output = output[:output.find('@')].strip()
            output = re.sub(' +', ' ', output)
            if log and batch_num % log_interval == 0:
                print([output, label])
            dist = editdistance.eval(output, label)
            cer += dist / max(len(output), len(label))

        return cer / batch_size

    def training_step(self, batch, batch_num):
        input_tensor, length_tensor, target_tensor = batch
        results, loss = self.forward(input_tensor, length_tensor, target_tensor)

        cer = self.decode(results, target_tensor, batch_num, log_interval=200, log=True)

        logs = {'train_loss': loss, 'train_cer': cer}
        return {'loss': loss, 'cer': cer, 'log': logs}

    def validation_step(self, batch, batch_num):
        input_tensor, lengths, target_tensor = batch
        results, loss = self.forward(input_tensor, lengths, target_tensor)
        cer = self.decode(results, target_tensor, batch_num, log_interval=10, log=True)

        return {
            'val_loss': loss,
            'val_cer': cer,
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_cer = np.mean([x['val_cer'] for x in outputs])

        if self.best_val_cer > avg_cer:
            self.best_val_cer = avg_cer
        logs = {
            'val_loss': avg_loss,
            'val_cer': avg_cer,
            'best_val_cer': self.best_val_cer
        }

        return {
            'val_loss': avg_loss,
            'val_cer': avg_cer,
            'log': logs,
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        train_data = LRS2Dataset(
            path=self.hparams.data,
            max_timesteps=self.max_timesteps,
            txtMaxLen=self.max_text_len,
            mode='train',
        )
        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_data = LRS2Dataset(
            path=self.hparams.data,
            max_timesteps=self.max_timesteps,
            txtMaxLen=self.max_text_len,
            mode='val',
        )
        val_loader = DataLoader(
            val_data, shuffle=False,
            batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers,
        )
        return val_loader


class Watch(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size):
        super(Watch, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder = Encoder()

    def forward(self, x, length):
        x = x.squeeze(dim=1)
        size = x.size()

        outputs = []
        for i in range(size[1] - 4):
            outputs.append(self.encoder(x[:, i:i+5, :, :]).unsqueeze(1))
        x = torch.cat(outputs, dim=1)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=length.view(-1).int(), batch_first=True, enforce_sorted=False)
        outputs, states = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0)
        return outputs, states[0]


class Listen(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(Listen, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(13, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, states = self.lstm(x)

        return (outputs, states[0])


class Spell(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size):
        super(Spell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedded = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers, batch_first=True)
        self.attentionVideo = Attention(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, input, hidden_state, cell_state, watch_outputs, context):
        input = self.embedded(input)
        concatenated = torch.cat([input, context], dim=2)
        output, (hidden_state, cell_state) = self.lstm(concatenated, (hidden_state, cell_state))
        context = self.attentionVideo(hidden_state[-1], watch_outputs)
        output = self.mlp(torch.cat([output, context], dim=2).squeeze(1)).unsqueeze(1)

        return output, hidden_state, cell_state, context


class Attention(nn.Module):
    def __init__(self, hidden_size, annotation_size):
        super(Attention, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size+annotation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, prev_hidden_state, annotations):
        batch_size, sequence_length, _ = annotations.size()

        prev_hidden_state = prev_hidden_state.repeat(sequence_length, 1, 1).transpose(0, 1)

        concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
        attn_energies = self.dense(concatenated).squeeze(2)
        alpha = F.softmax(attn_energies, dim=1).unsqueeze(1)
        context = alpha.bmm(annotations)

        return context


class Encoder(nn.Module):
    '''modified VGG-M
    '''

    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 96, (7, 7), (2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        )

        self.fc = nn.Linear(4608, 512)

    def forward(self, x):
        # x = self.encoder(x).view(x.size(0), -1)
        # return self.fc(x)
        return self.fc(checkpoint_sequential(self.encoder, len(self.encoder), x).view(x.size(0), -1))
