# PyTorch Trainer

Lightweight wrapper around PyTorch. Removes boilerplate code to focus on the important parts.

## Example
```python
import os

import torch
import torchvision.transforms as transforms
from early_stopping import EarlyStopping
from model_checkpoint import ModelCheckpoint
from module import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from trainer import Trainer


class MNISTModel(Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch):
        x, y = batch
        output = self.forward(x)
        return {'val_loss': F.cross_entropy(output, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def val_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)


checkpoint_callback = ModelCheckpoint(
    directory='./checkpoints',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=5,
    mode='min'
)

model = MNISTModel()
trainer = Trainer(
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=early_stop_callback,
)
trainer.fit(model)
```

Inspired by [PyTorch Lightning](https://github.com/williamFalcon/pytorch-lightning)
