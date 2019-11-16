import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_trainer import Module, data_loader
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
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

        self.epoch = 0

    def forward(self, x):
        x = self.frontend(x)
        x = self.resnet(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_num):
        frames = batch['frames']
        labels = batch['label']
        output = self.forward(frames)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_step(self, batch, batch_num):
        frames = batch['frames']
        labels = batch['label']
        words = batch['word']
        output = self.forward(frames)
        loss = self.loss(output, labels.squeeze(1))
        acc = accuracy(output, labels)
        sums = torch.sum(output, dim=1)
        _, predicted = sums.max(dim=1)
        return {
            'val_loss': loss,
            'val_acc': acc,
            'predictions': predicted,
            'labels': labels.squeeze(dim=1),
            'words': words,
        }

    def validation_end(self, outputs):
        predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        words = np.concatenate([x['words'] for x in outputs])
        self.confusion_matrix(labels, predictions, words)

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        if self.best_val_acc < avg_acc:
            self.best_val_acc = avg_acc
        logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'best_val_acc': self.best_val_acc
        }

        self.epoch += 1
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'log': logs,
        }

    def confusion_matrix(self, label, prediction, words, normalize=True):
        classes = unique_labels(label, prediction)
        cm = confusion_matrix(prediction, label)
        cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
        ax.set_ylabel("Label")
        ax.set_xlabel("Predicted")
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("Word Confusion Matrix")

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        directory = "data/viz/lrw"
        os.makedirs(directory, exist_ok=True)
        path = f"{directory}/cm_seed_{self.hparams.seed}_epoch_{self.epoch}.png"
        plt.savefig(path)
        self.logger.save_file(path)
        plt.clf()
        plt.close()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

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
