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

from src.checkpoint import create_checkpoint, load_checkpoint
from src.data.hdf5 import HDF5Dataset
from src.data.lrw import LRWDataset
from src.models.model import Model

epochs = 50
learning_rate = 1e-4
num_classes = 10

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument("--checkpoint_dir", type=str, default='data/models')
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--tensorboard_logdir", type=str, default='data/tensorboard')
parser.add_argument("--batch_size", type=int, default=24)
args = parser.parse_args()

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
data_path = args.data
checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + current_time + ".pkl")
checkpoint = args.checkpoint
tensorboard_logdir = args.tensorboard_logdir
batch_size = args.batch_size

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = HDF5Dataset(path=data_path)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(HDF5Dataset(path=data_path, table='val'), shuffle=False, batch_size=batch_size * 2)
samples = len(train_data)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logdir, current_time))
model = Model(num_classes=num_classes, pretrained_resnet=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if checkpoint != None:
    load_checkpoint(model, optimizer, checkpoint)


def train(epoch, start_time):
    model.train()
    criterion = model.loss
    batch_times, load_times, accuracies = np.array([]), np.array([]), np.array([])
    loader = iter(train_loader)
    for step in range(1, len(train_loader) + 1):
        batch_start = time.time()
        batch = next(loader)
        load_times = np.append(load_times, time.time() - batch_start)

        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        output = model(inputs)
        loss = criterion(output, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc = accuracy(output, labels)
        acc_top = accuracy_topk(output, labels, k=5)
        accuracies = np.append(accuracies, acc)

        batch_times = np.append(batch_times, time.time() - batch_start)
        global_step = ((epoch * samples) // batch_size) + step
        writer.add_scalar("train_loss", loss, global_step=global_step)
        writer.add_scalar("train_acc", acc, global_step=global_step)
        if step % 50 == 0:
            duration = time.time() - start_time
            epoch_samples = batch_size * step
            samples_processed = (epoch * samples) + epoch_samples
            total_samples = epochs * samples
            remaining_time = (total_samples - samples_processed) * (duration / samples_processed)
            print("Epoch: %d, %d/%d samples, Loss: %.2f, Time per sample: %.2fms, Load sample: %.2fms, Train acc: %.5f, Elapsed time: %s, Remaining time: %s" % (
                epoch + 1,
                epoch_samples,
                samples,
                loss,
                (np.mean(batch_times) * 1000) / batch_size,
                (np.mean(load_times) * 1000) / batch_size,
                np.mean(accuracies),
                time.strftime("%H:%M:%S", time.gmtime(duration)),
                time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
            ))
            batch_times, load_times, accuracies = np.array([]), np.array([]), np.array([])
        if step % 500 == 0:
            create_checkpoint(model, optimizer, checkpoint_path)
            print("Saved checkpoint at step %d" % global_step)


def validate(epoch):
    model.eval()
    criterion = model.loss
    accuracies, losses = np.array([]), np.array([])
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            output = model(inputs)
            loss = criterion(output, labels.squeeze(1))

            acc = accuracy(output, labels)
            losses = np.append(losses, loss.item())
            accuracies = np.append(accuracies, acc)

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    global_step = (epoch + 1) * samples
    writer.add_scalar("val_loss", avg_loss, global_step=global_step)
    writer.add_scalar("val_acc", avg_acc, global_step=global_step)
    print("Val loss: %.3f, Val acc %.5f" % (avg_loss, avg_acc))


def accuracy(output, labels):
    sums = torch.sum(output, dim=1)
    _, predicted = sums.max(dim=1)
    correct = (predicted == labels.squeeze(1)).sum().item()
    return correct / output.shape[0]


def accuracy_topk(outputs, labels, k=10):
    sums = torch.sum(outputs, dim=1)
    _, predicted = sums.topk(k=k, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / outputs.shape[0]


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters: %d" % trainable_params)
start_time = time.time()
for epoch in range(epochs):
    train(epoch, start_time)
    validate(epoch)
