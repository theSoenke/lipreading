import difflib
import os
import random
import re

import ctcdecode
import editdistance
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_trainer import Module
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader

from src.data.lrs2 import LRS2Dataset
from src.models.resnet import ResNetModel


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, smoothing, vocab_size, ignore_index):
        assert 0.0 < smoothing <= 1.0
        self.ignore_index = ignore_index
        super().__init__()

        smoothing_value = smoothing / (vocab_size - 1)
        one_hot = torch.full((vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        output = output.log_softmax(dim=1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class LRS2ResnetAttn(Module):
    def __init__(self, hparams, in_channels=1, pretrain=False):
        super().__init__()
        self.hparams = hparams
        self.in_channels = in_channels
        self.pretrain = pretrain
        self.max_timesteps = 155
        self.max_text_len = 100
        self.pretrain_words = 0

        self.teacher_forcing_ratio = 1.0
        self.min_teacher_forcing_ratio = 0.75

        dataset = self.train_dataloader().dataset
        self.int2char = dataset.int2char
        self.char2int = dataset.char2int
        self.dictionary = dataset.dictionary

        self.frontend = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNetModel(
            layers=hparams.resnet,
            output_dim=512,
            pretrained=hparams.pretrained,
            large_input=False
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=3,
            batch_first=True,
            bidirectional=False,
        )
        num_characters = len(dataset.char_list)
        self.spell = AttentionDecoder(3, 512, num_characters)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.char2int['<pad>'])
        self.criterion = LabelSmoothingLoss(smoothing=0.1, vocab_size=num_characters, ignore_index=self.char2int['<pad>'])

        self.vocab_list = dataset.char_list
        self.vocab_list[self.vocab_list.index("<pad>")] = "_"
        self.vocab_list[self.vocab_list.index("<eos>")] = "@"
        self.vocab_list[self.vocab_list.index("<sos>")] = "$"
        self.decoder = ctcdecode.CTCBeamDecoder(
            self.vocab_list,
            model_path=hparams.lm_path,
            alpha=1.0,
            beta=1.0,
            cutoff_top_n=50,
            cutoff_prob=0.99,
            beam_width=200,
            blank_id=self.vocab_list.index("_"),
        )

        self.best_val_cer = 1.0
        self.best_val_wer = 1.0

    def forward(self, x, lengths, target_tensor, enable_teacher=True):
        x = self.frontend(x)
        x = self.resnet(x)
        x = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        x, states = self.lstm(x)
        watch_outputs, _ = pad_packed_sequence(x, batch_first=True)
        spell_hidden = states[0]

        device = self.trainer.device
        decoder_input = torch.tensor([self.char2int['<sos>']], device=device).repeat(watch_outputs.size(0), 1)
        cell_state = torch.zeros_like(spell_hidden, device=device)
        context = torch.zeros(watch_outputs.size(0), 1, spell_hidden.size(2), device=device)

        loss = 0
        results = []
        max_length = target_tensor.size(1)
        decoder_attentions = []
        for i in range(max_length):
            use_teacher_forcing = True if enable_teacher and random.random() < self.teacher_forcing_ratio else False
            decoder_output, spell_hidden, cell_state, context, attn_weights = self.spell(
                decoder_input, spell_hidden, cell_state, watch_outputs, context)
            _, topi = decoder_output.topk(1, dim=2)
            decoder_attentions.append(attn_weights.squeeze(dim=1))
            if use_teacher_forcing:
                decoder_input = target_tensor[:, i].long().unsqueeze(dim=1)
            else:
                decoder_input = topi.squeeze(dim=1).detach()
            results.append(decoder_output.squeeze(dim=1))
            loss += self.criterion(decoder_output.squeeze(dim=1), target_tensor[:, i].long())

        decoder_attentions = torch.stack(decoder_attentions, dim=1)
        results = torch.stack(results, dim=1).softmax(dim=2)
        return loss / max_length, results, decoder_attentions

    def decode(self, label_tokens, target_tokens, use_dictionary=False):
        label, output = '', ''
        for index in range(len(label_tokens)):
            label += self.int2char[int(label_tokens[index])]
        for index in range(len(target_tokens)):
            output += self.int2char[int(target_tokens[index])]
        label = label.replace('<pad>', ' ').replace('<eos>', '@')
        label = label[:label.find("@")]
        output = output.replace('<eos>', '@').replace('<pad>', '&').replace('<sos>', '&')
        output = output[:output.find('@')].strip()
        output = re.sub(' +', ' ', output)
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)  # remove characters that are repeated more than 3 times
        output = pattern.sub(r"\1", output)

        output_words, label_words = output.split(" "), label.split(" ")
        if use_dictionary:
            for i, word in enumerate(output_words):
                if word not in self.dictionary:
                    closest_words = difflib.get_close_matches(word, self.dictionary, cutoff=0.9)
                    if len(closest_words) > 0:
                        output_words[i] = closest_words[0]

            output = ' '.join(output_words)

        cer = editdistance.eval(output, label) / max(len(output), len(label))
        wer = editdistance.eval(output_words, label_words) / max(len(output_words), len(label_words))

        return label, output, cer, wer

    def greedy_decode(self, results, target):
        _, results = results.topk(1, dim=2)
        results = results.squeeze(dim=2)
        cer_sum, wer_sum = 0, 0
        batch_size = results.size(0)
        sentences = []
        for batch in range(batch_size):
            label, output, cer, wer = self.decode(target[batch], results[batch])
            sentences.append([label, output])
            cer_sum += cer
            wer_sum += wer

        return cer_sum / batch_size, wer_sum / batch_size, sentences

    def beam_decode(self, results, target):
        beam_results, beam_scores, timesteps, out_seq_len = self.decoder.decode(results)
        batch_size = results.size(0)
        cer_sum, wer_sum = 0, 0
        sentences = []
        for batch in range(batch_size):
            seq_len = out_seq_len[batch][0]
            tokens = beam_results[batch][0][:seq_len]  # select output with best score
            label, output, cer, wer = self.decode(target[batch], tokens)
            sentences.append([label, output])
            cer_sum += cer
            wer_sum += wer

        return cer_sum / batch_size, wer_sum / batch_size, sentences

    def training_step(self, batch, batch_num):
        frames, input_lengths, target = batch
        loss, results, _ = self.forward(frames, input_lengths, target)
        cer, wer, sentences = self.greedy_decode(results, target)

        logs = {'train_loss': loss, 'train_cer': cer, 'train_wer': wer}
        return {'loss': loss, 'cer': cer, 'teacher_forcing': self.teacher_forcing_ratio, 'log': logs}

    def validation_step(self, batch, batch_num):
        frames, input_lengths, target = batch
        loss, results, attn_weights = self.forward(frames, input_lengths, target, enable_teacher=False)
        # self.save_attention(results, target, input_lengths, attn_weights)
        cer, wer, sentences_greedy = self.greedy_decode(results, target)
        beam_cer, beam_wer, sentences_beam = self.beam_decode(results, target)

        batch_size = results.size(0)
        if batch_num % 10 == 0:
            for i in range(batch_size):
                print(f"Label: {sentences_greedy[i][0]}\nGreedy: {sentences_greedy[i][1]}\nBeam: {sentences_beam[i][1]}\n")

        return {
            'val_loss': loss,
            'val_cer': cer,
            'val_wer': wer,
            'val_beam_cer': beam_cer,
            'val_beam_wer': beam_wer,
        }

    def validation_end(self, outputs):
        cer = np.mean([x['val_cer'] for x in outputs])
        wer = np.mean([x['val_wer'] for x in outputs])
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        beam_cer = np.mean([x['val_beam_cer'] for x in outputs])
        beam_wer = np.mean([x['val_beam_wer'] for x in outputs])

        if self.best_val_cer > cer:
            self.best_val_cer = cer
        if self.best_val_wer > wer:
            self.best_val_wer = wer
        logs = {
            'val_loss': loss,
            'val_cer': cer,
            'val_wer': wer,
            'val_beam_cer': beam_cer,
            'val_beam_wer': beam_wer,
            'best_val_cer': self.best_val_cer,
            'best_val_wer': self.best_val_wer,
        }

        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step(loss)
            for param_group in self.trainer.optimizer.param_groups:
                logs['lr'] = param_group['lr']

        return {
            'val_loss': loss,
            'val_cer': cer,
            'val_wer': wer,
            'log': logs,
        }

    def save_attention(self, results, target, input_lengths, attn_weights):
        _, _, sentences = self.greedy_decode(results, target)
        batch_size = results.size(0)
        for i in range(batch_size):
            label, output = sentences[i]
            self.plot_attention(label, output, attn_weights[i][:len(output), :input_lengths[i]].cpu())

    def plot_attention(self, input_sentence, output_sentence, attentions):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy())
        fig.colorbar(cax)

        ax.set_ylabel(input_sentence)
        ax.set_xlabel(output_sentence)

        directory = "data/viz/lrs2/attention"
        os.makedirs(directory, exist_ok=True)
        path = f"{directory}/{input_sentence}.pdf"
        plt.savefig(path)
        plt.clf()
        plt.close()

    def on_epoch_start(self, epoch):
        decay_rate = (1.0 - self.min_teacher_forcing_ratio) / (self.hparams.epochs - 1)
        self.teacher_forcing_ratio = 1.0 - (epoch * decay_rate)
        self.trainer.logger.log_metrics({'teacher_forcing': self.teacher_forcing_ratio})
        print(f"Use teacher forcing ratio: {self.teacher_forcing_ratio}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True,
        )

        return optimizer, scheduler

    def train_dataloader(self):
        train_data = LRS2Dataset(
            path=self.hparams.data,
            max_text_len=self.max_text_len,
            mode='train',
            max_timesteps=self.max_timesteps,
            pretrain_words=self.pretrain_words,
            pretrain=self.pretrain,
            augmentations=True,
        )
        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_data = LRS2Dataset(
            path=self.hparams.data,
            mode='val',
            max_timesteps=112,
            max_text_len=100,
            pretrain_words=0,
            pretrain=False,
        )
        val_loader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=self.hparams.batch_size * 2,
            num_workers=self.hparams.workers,
        )
        return val_loader


class AttentionDecoder(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedded = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.attention = Attention(hidden_size, hidden_size)
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
        context, attn_weights = self.attention(hidden_state[-1], watch_outputs)
        output = self.mlp(torch.cat([output, context], dim=2).squeeze(dim=1)).unsqueeze(dim=1)
        output = F.log_softmax(output, dim=2)
        return output, hidden_state, cell_state, context, attn_weights


class Attention(nn.Module):
    def __init__(self, hidden_size, annotation_size):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size+annotation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, prev_hidden_state, annotations):
        batch_size, sequence_length, _ = annotations.size()
        prev_hidden_state = prev_hidden_state.repeat(sequence_length, 1, 1).transpose(0, 1)

        concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
        attn_energies = self.dense(concatenated).squeeze(dim=2)
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(dim=1)
        context = attn_weights.bmm(annotations)

        return context, attn_weights
