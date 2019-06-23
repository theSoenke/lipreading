import argparse
import datetime
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.lrw import LRWDataset
from src.models.model import Model

epochs = 10
learning_rate = 1e-3
batch_size = 24

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument("--checkpoint", type=str, default='data/models/checkpoint.pkl')
parser.add_argument("--tensorboard_logdir", type=str, default='data/tensorboard')
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

data_path = args.data
checkpoint_path = args.checkpoint
tensorboard_logdir = args.tensorboard_logdir
workers = args.workers

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = DataLoader(LRWDataset(directory=data_path, mode='train'), shuffle=True, batch_size=batch_size, num_workers=workers)
samples = len(train_data) * batch_size

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logdir, current_time))
model = Model(num_classes=500, pretrained_resnet=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch, start_time):
    criterion = model.loss
    step = (epoch * samples) // batch_size
    batch_times = np.array([])
    load_times = np.array([])
    batch_num = len(train_data)
    loader = iter(train_data)
    for step in range(1, batch_num + 1):
        batch_start = time.time()
        batch = next(loader)
        load_times = np.append(load_times, time.time() - batch_start)

        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        pred = model(inputs)
        loss = criterion(pred, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_times = np.append(batch_times, time.time() - batch_start)
        writer.add_scalar("train_loss", loss, global_step=step)
        if step % 50 == 0:
            epoch_samples = batch_size * step
            duration = time.time() - start_time
            time_left = (samples - epoch_samples) * (duration / epoch_samples)
            print("%d/%d samples, Loss: %f, Time per sample: %fms, Load sample: %fms, Elapsed time: %s, Remaining time: %s" % (
                epoch_samples,
                samples,
                loss,
                (np.mean(batch_times) * 1000) / batch_size,
                (np.mean(load_times) * 1000) / batch_size,
                time.strftime("%H:%M:%S", time.gmtime(duration)),
                time.strftime("%H:%M:%S", time.gmtime(time_left)),
            ))
            batch_times = np.array([])
            load_times = np.array([])
        if step % 500 == 0:
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved checkpoint at step %d" % step)


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters: %d" % trainable_params)
start_time = time.time()
for epoch in range(epochs):
    model.train()
    train(epoch, start_time)
