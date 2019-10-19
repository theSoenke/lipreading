import os

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint
from src.data.lrw import LRWDataset
from src.models.attention import Attention
from src.models.lrw_model import LRWModel
from src.trainer.module import Module


class ExpertModel(Module):
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
        output = (left_flat + center_flat + right_flat).view(x.size(0), 29, 10)

        return output, attn

    def training_step(self, batch, batch_nb):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output, attn = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = LRWModel.accuracy(output, labels)
        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_step(self, batch, batch_nb):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output, attn = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = LRWModel.accuracy(output, labels)

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

    def test_step(self, batch, batch_nb):
        frames = batch['frames']
        labels = batch['label']
        yaws = batch['yaw']

        output, _ = self.forward(frames, yaws)
        loss = self.loss(output, labels.squeeze(1))
        acc = LRWModel.accuracy(output, labels)
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

        # left_buckets, right_buckets = {}, {}
        # for i in range(len(yaws)):
        #     degree = int(yaws[i])
        #     if degree in left_buckets:
        #         left_buckets[degree].append(left_attn[i])
        #     else:
        #         left_buckets[degree] = [left_attn[i]]

        #     if degree in right_buckets:
        #         right_buckets[degree].append(right_attn[i])
        #     else:
        #         right_buckets[degree] = [right_attn[i]]

        size = 40
        plt.scatter(yaws, left_attn, s=size, label='left expert')
        plt.scatter(yaws, center_attn, s=size, label='center expert')
        plt.scatter(yaws, right_attn, s=size, label='right expert')
        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.xlabel('Degree')
        plt.ylabel('Attention')

        plt.savefig(f"attention_seed_{self.hparams.seed}_epoch_{self.epoch}.png")
        plt.clf()
        self.epoch += 1

    def configure_optimizers(self):
        return optim.Adam(self.attention.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        train_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, seed=self.hparams.seed)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True)
        return train_loader, len(train_data)

    def val_dataloader(self):
        val_data = LRWDataset(path=self.hparams.data, num_words=self.hparams.words, mode='val', seed=self.hparams.seed)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=self.hparams.batch_size * 2, num_workers=self.hparams.workers)
        return val_loader, len(val_data)
