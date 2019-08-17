import argparse
import datetime
import os
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
from src.data.grid import GRIDDataset
from src.data.ouluvs2 import OuluVS2Dataset
from src.data.utils import ctc_collate
from src.models.ctc_decoder import Decoder
from src.models.lipnet import LipNet

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/grid')
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--tensorboard_logdir", type=str, default='data/tensorboard')
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--words", type=int, default=10)
parser.add_argument("--resnet", type=int, default=18)
parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
log_interval = args.log_interval

torch.manual_seed(42)
np.random.seed(42)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = OuluVS2Dataset(path=args.data, augmentation=True)
train_loader = DataLoader(
    train_data,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=ctc_collate,
    num_workers=args.workers,
    pin_memory=True,
)
val_data = GRIDDataset(path=args.data, mode='val')
val_loader = DataLoader(
    val_data,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=ctc_collate,
    num_workers=args.workers,
)
samples = len(train_data)
os.makedirs(args.checkpoint_dir, exist_ok=True)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_logdir, current_time))
pretrained = False if args.checkpoint != None else args.pretrained
model = LipNet(vocab_size=len(train_data.vocab)).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
decoder = Decoder(train_data.vocab)
criterion = nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

wandb.init(project="grid")
wandb.config.update(args)
wandb.watch(model)

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
        loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
        loss = loss_all.mean()

        weight = torch.ones_like(loss_all)
        dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]
        logits.backward(dlogits)
        optimizer.step()
        optimizer.zero_grad()

        batch_times = np.append(batch_times, time.time() - batch_start)
        samples_processed += len(x)
        global_step = ((epoch * samples) // batch_size) + step
        wandb.log({"train_loss": loss})
        if step % log_interval == 0:
            duration = time.time() - start_time
            total_samples_processed = (epoch * samples) + samples_processed
            total_samples = epochs * samples
            remaining_time = (total_samples - total_samples_processed) * (duration / total_samples_processed)
            print(
                f"Epoch: [{epoch + 1}/{epochs}], "
                + f"{samples_processed}/{samples} samples, "
                + f"Loss: {loss:.2f}, "
                + f"Time per sample: {((np.mean(batch_times) * 1000) / batch_size) / log_interval:.3f}ms, "
                + f"Load sample: {((np.mean(load_times) * 1000) / batch_size) / log_interval:.3f}ms, "
                + f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(duration))}, "
                + f"Remaining time: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")
            batch_times, load_times, accuracies = np.array([]), np.array([]), np.array([])


@torch.no_grad()
def validate(epoch):
    model.eval()
    losses = np.array([]), np.array([])
    predictions, ground_truth = [], []

    for batch in val_loader:
        x, y, lengths, y_lengths, idx = batch
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
        loss = loss_all.mean()

        decoded, gt = predict(x.size(0), logits, y, lengths, y_lengths, n_show=5, mode='greedy')
        predictions.extend(decoded)
        ground_truth.extend(gt)
        losses = np.append(losses, loss.item())

    wer = decoder.wer_batch(predictions, ground_truth)
    cer = decoder.cer_batch(predictions, ground_truth)
    avg_loss = np.mean(losses)
    global_step = (epoch + 1) * samples
    writer.add_scalar("val_loss", avg_loss, global_step=global_step)
    writer.add_scalar("val_wer", wer, global_step=global_step)
    writer.add_scalar("val_cer", cer, global_step=global_step)
    wandb.log({"val_loss": avg_loss, 'val_wer': wer, 'val_cer': cer})
    print(f"val_loss: {avg_loss:.3f}, val_wer: {wer:.5f}, val_cer: {cer:.5f}")

    return wer, cer


def predict(batch_size, logits, y, lengths, y_lengths, n_show=5, mode='greedy', log=False):
    if mode == 'greedy':
        decoded = decoder.decode_greedy(logits, lengths)
    elif mode == 'beam':
        decoded = decoder.decode_beam(logits, lengths)

    cursor = 0
    gt = []
    n = min(n_show, logits.size(1))
    for b in range(batch_size):
        y_str = ''.join([val_data.vocab[ch - 1] for ch in y[cursor: cursor + y_lengths[b]]])
        gt.append(y_str)
        cursor += y_lengths[b]
        if log and b < n:
            print(f'Test seq {b + 1}: {y_str}; pred_{mode}: {decoded[b]}')

    return decoded, gt


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
start_time = time.time()
best_val_acc = 0
for epoch in range(epochs):
    train(epoch, start_time)
    validate(epoch)
    checkpoint_name = f"checkpoint_{epoch}_{current_time}.pkl"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    create_checkpoint(checkpoint_path, model)

wandb.config.parameters = trainable_params
wandb.save(checkpoint_path)
