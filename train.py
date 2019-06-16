import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.lrw import LRWDataset
from src.models.model import Model

epochs = 10
learning_rate = 1e-3
batch_size = 18

parser = argparse.ArgumentParser()
parser.add_argument('--data')
args = parser.parse_args()
data_path = args.data

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = DataLoader(LRWDataset(directory=data_path, mode='train'), shuffle=True, batch_size=batch_size, num_workers=4)

model = Model(num_classes=500).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    criterion = model.loss
    for batch in train_data:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        pred = model(inputs)
        loss = criterion(pred, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


for epoch in range(epochs):
    model.train()
    train(epoch)
