import torch
from early_stopping import EarlyStopping
from model_checkpoint import ModelCheckpoint
from module import Module
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from trainer import Trainer
from wandb_logger import WandbLogger


class MNISTModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch):
        x, target = batch

        output = self.forward(x)
        loss = F.nll_loss(output, target)

        logger = {'train_loss': loss}
        return {'loss': loss, 'log': logger}

    def validation_step(self, batch):
        x, target = batch

        output = self.forward(x)
        loss = F.nll_loss(output, target)

        logger = {'val_loss': loss}
        return {'val_loss': loss, 'log': logger}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {
            'val_loss': avg_loss,
        }
        return {
            'val_loss': avg_loss,
            'log': logs,
        }

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

        return loader

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        dataset = MNIST(os.getcwd(), train=False, transform=transform, download=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )

        return loader


if __name__ == "__main__":
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
    logger = WandbLogger("test", model)
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger
    )
    trainer.fit(model)
