import argparse
import datetime
import os
import pdb
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb
from src.checkpoint import create_checkpoint, load_checkpoint
from src.data.grid import GRIDDataset, ctc_collate
from src.models.ctc_decoder import Decoder
from src.models.lipnet import LipNet

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints')
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--tensorboard_logdir", type=str, default='data/tensorboard')
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=156)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--words", type=int, default=10)
parser.add_argument("--resnet", type=int, default=18)
parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--log_interval", type=int, default=50)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
log_interval = args.log_interval

torch.manual_seed(42)
np.random.seed(42)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = GRIDDataset(path=args.data)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,  collate_fn=ctc_collate, pin_memory=True)
val_loader = DataLoader(GRIDDataset(path=args.data, mode='val'), shuffle=False, batch_size=batch_size * 2, collate_fn=ctc_collate)
samples = len(train_data)
os.makedirs(args.checkpoint_dir, exist_ok=True)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logdir, current_time))
pretrained = False if args.checkpoint != None else args.pretrained
model = LipNet(vocab_size=len(train_data.vocab)).to(device)
decoder = Decoder(train_data.vocab)
crit = nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
wandb.init(project="grid")
wandb.config.update(args)
wandb.watch(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.checkpoint != None:
    load_checkpoint(args.checkpoint, model, optimizer=None)


def train(epoch, start_time):
    model.train()
    batch_times, load_times, accuracies = np.array([]), np.array([]), np.array([])
    samples_processed = 0
    loader = iter(train_loader)
    for step in range(1, len(train_loader) + 1):
        batch_start = time.time()
        batch = next(loader)
        load_times = np.append(load_times, time.time() - batch_start)

        x, y, lengths, y_lengths, idx = batch
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss_all = crit(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
        loss = loss_all.mean()

        weight = torch.ones_like(loss_all)
        dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]
        logits.backward(dlogits)
        optimizer.step()
        optimizer.zero_grad()

        acc = 0  # TODO
        accuracies = np.append(accuracies, acc)

        batch_times = np.append(batch_times, time.time() - batch_start)
        samples_processed += len(x)
        global_step = ((epoch * samples) // batch_size) + step
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


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
start_time = time.time()
best_val_acc = 0
for epoch in range(epochs):
    train(epoch, start_time)

wandb.config.parameters = trainable_params
# wandb.save(checkpoint_path)
