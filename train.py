import argparse
import datetime
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.functional import F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb
from src.checkpoint import create_checkpoint, load_checkpoint
from src.data.lrw import LRWDataset
from src.models.attention import *
from src.models.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints')
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--checkpoint2", type=str)
parser.add_argument("--checkpoint3", type=str)
parser.add_argument("--tensorboard_logdir", type=str, default='data/tensorboard')
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--words", type=int, default=10)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--resnet", type=int, default=18)
parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
log_interval = args.log_interval
pretrained = False if args.checkpoint != None else args.pretrained
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
query = None  # (-20, 20)
train_data = LRWDataset(path=args.data, num_words=args.words, query=query, seed=args.seed)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
val_data = LRWDataset(path=args.data, num_words=args.words, mode='val', query=query, seed=args.seed)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size * 2, num_workers=args.workers)
samples = len(train_data)

writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logdir, current_time))
model = Model(num_classes=args.words,  resnet_layers=args.resnet, resnet_pretrained=pretrained).to(device)
load_checkpoint(args.checkpoint, model, optimizer=None)
wandb.init(project="lipreading")
wandb.config.update(args)
wandb.watch(model)

for param in model.parameters():
    param.requires_grad = False

num_experts = 3
attention = Attention(attention_dim=40, num_experts=num_experts).to(device)

optimizer = optim.Adam(attention.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model2 = Model(num_classes=args.words,  resnet_layers=args.resnet, resnet_pretrained=pretrained).to(device)
load_checkpoint(args.checkpoint2, model2, optimizer=None)
model3 = Model(num_classes=args.words,  resnet_layers=args.resnet, resnet_pretrained=pretrained).to(device)
load_checkpoint(args.checkpoint3, model3, optimizer=None)


def train(epoch, start_time):
    model.train()
    criterion = model2.loss
    batch_times, load_times, accuracies = np.array([]), np.array([]), np.array([])
    loader = iter(train_loader)
    samples_processed = 0
    for step in range(1, len(train_loader) + 1):
        batch_start = time.time()
        batch = next(loader)
        load_times = np.append(load_times, time.time() - batch_start)

        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        yaws = batch['yaw']
        yaws = torch.FloatTensor([float(yaw) for yaw in yaws]).unsqueeze(dim=1).to(device)

        output1 = model.front_pass(frames)
        output2 = model2.front_pass(frames)
        output3 = model3.front_pass(frames)

        context = attention(yaws)
        expert_attn = context.split(split_size=1, dim=1)
        # output1_flat = output1.view(frames.size(0), -1)
        # output2_flat = output2.view(frames.size(0), -1)
        # output3_flat = output3.view(frames.size(0), -1)
        # output1_flat = output1_flat * expert_attn[0]
        # output2_flat = output2_flat * expert_attn[1]
        # output3_flat = output3_flat * expert_attn[2]
        # output = (output1_flat + output2_flat + output3_flat).view(frames.size(0), 29, 256)
        output1 = model(output1)
        output2 = model2(output2)
        output3 = model3(output3)

        output1_flat = output1.view(frames.size(0), -1)
        output2_flat = output2.view(frames.size(0), -1)
        output3_flat = output3.view(frames.size(0), -1)
        output1_flat = output1_flat * expert_attn[0]
        output2_flat = output2_flat * expert_attn[1]
        output3_flat = output3_flat * expert_attn[2]

        output = (output1_flat + output2_flat + output3_flat).view(frames.size(0), 29, 10)

        loss = criterion(output, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc = accuracy(output, labels)
        accuracies = np.append(accuracies, acc)

        batch_times = np.append(batch_times, time.time() - batch_start)
        samples_processed += len(labels)
        global_step = ((epoch * samples) // batch_size) + step
        writer.add_scalar("train_loss", loss, global_step=global_step)
        writer.add_scalar("train_acc", acc, global_step=global_step)
        wandb.log({"train_acc": acc, "train_loss": loss})
        if step % log_interval == 0:
            duration = time.time() - start_time
            total_samples_processed = (epoch * samples) + samples_processed
            total_samples = epochs * samples
            remaining_time = (total_samples - total_samples_processed) * (duration / total_samples_processed)
            print(
                f"Epoch: [{epoch + 1}/{epochs}], "
                + f"{samples_processed}/{samples} samples, "
                + f"Loss: {loss:.2f}, "
                + f"Time per sample: {((np.mean(batch_times) * 1000) / batch_size) / log_interval:.2f}ms, "
                + f"Load sample: {((np.mean(load_times) * 1000) / batch_size) / log_interval:.2f}ms, "
                + f"Train acc: {np.mean(accuracies):.4f}, "
                + f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(duration))}, "
                + f"Remaining time: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")
            batch_times, load_times, accuracies = np.array([]), np.array([]), np.array([])


@torch.no_grad()
def validate(epoch):
    model.eval()
    criterion = model2.loss
    accuracies, losses = np.array([]), np.array([])
    lefts, centers, rights, degrees = np.array([]), np.array([]), np.array([]), np.array([])
    for batch in val_loader:
        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        yaws = batch['yaw']
        yaws = torch.FloatTensor([float(yaw) for yaw in yaws]).unsqueeze(dim=1).to(device)

        output1 = model.front_pass(frames)
        output2 = model2.front_pass(frames)
        output3 = model3.front_pass(frames)

        output1 = model(output1)
        output2 = model2(output2)
        output3 = model3(output3)

        context = attention(yaws)
        expert_attn = context.split(split_size=1, dim=1)
        output1_flat = output1.view(frames.size(0), -1)
        output2_flat = output2.view(frames.size(0), -1)
        output3_flat = output3.view(frames.size(0), -1)
        output1_flat = output1_flat * expert_attn[0]
        output2_flat = output2_flat * expert_attn[1]
        output3_flat = output3_flat * expert_attn[2]

        output = (output1_flat + output2_flat + output3_flat).view(frames.size(0), 29, 10)
        for i, yaw in enumerate(yaws):
            yaw = yaw.item()
            left = context[i][0].item()
            center = context[i][1].item()
            right = context[i][2].item()
            lefts = np.append(lefts, left)
            centers = np.append(centers, center)
            rights = np.append(rights, right)
            degrees = np.append(degrees, yaw)
            print(f"Degree: {yaw}, Left: {left:.3f}, Center: {center:.3f}, Right: {right:.3f}")

        # expert_attn = context.split(split_size=1, dim=1)
        # output1_flat = output1_flat * expert_attn[0]
        # output2_flat = output2_flat * expert_attn[1]
        # output3_flat = output3_flat * expert_attn[2]
        # output = (output1_flat + output2_flat + output3_flat).view(frames.size(0), 29, 256)

        # output = model2(output)
        loss = criterion(output, labels.squeeze(1))

        acc = accuracy(output, labels)
        losses = np.append(losses, loss.item())
        accuracies = np.append(accuracies, acc)

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    global_step = (epoch + 1) * samples
    writer.add_scalar("val_loss", avg_loss, global_step=global_step)
    writer.add_scalar("val_acc", avg_acc, global_step=global_step)
    wandb.log({"val_acc": avg_acc, "val_loss": avg_loss})
    print(f"val_loss: {avg_loss:.3f}, val_acc {avg_acc:.5f}")

    width = 0.4
    indices = np.arange(len(lefts))
    plt.bar(indices, lefts, width, color='r')
    plt.bar(indices, centers, width, bottom=lefts, color='b')
    plt.bar(indices, rights, width, bottom=lefts + centers, color='g')
    # plt.bar(indices, degrees, width=width, bottom=lefts + centers + rights)
    plt.show()

    return avg_acc


def accuracy(output, labels):
    sums = torch.sum(output, dim=1)
    _, predicted = sums.max(dim=1)
    correct = (predicted == labels.squeeze(dim=1)).sum().item()
    return correct / output.shape[0]


def accuracy_topk(outputs, labels, k=10):
    sums = torch.sum(outputs, dim=1)
    _, predicted = sums.topk(k=k, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / outputs.shape[0]


trainable_params = sum(p.numel() for p in attention.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
start_time = time.time()
best_val_acc = 0
os.makedirs(args.checkpoint_dir, exist_ok=True)
val_acc = validate(0)  # log initial val acc
wandb.log({"best_val_acc": best_val_acc})
for epoch in range(epochs):
    train(epoch, start_time)
    val_acc = validate(epoch)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        wandb.log({"best_val_acc": best_val_acc})
        checkpoint_name = "checkpoint_%d_val_acc_%.5f_%s.pkl" % (epoch, val_acc, current_time)
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
        create_checkpoint(checkpoint_path, model)
        print(f"Saved checkpoint: {checkpoint_path}")

wandb.config.parameters = trainable_params
wandb.save(checkpoint_path)
