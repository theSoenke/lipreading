import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.hdf5 import HDF5Dataset
from src.models.model import Model
from src.data.lrw import build_word_list


@torch.no_grad()
def evaluate(model, dataloader, num_words, directory):
    words = build_word_list(directory=directory, num_words=num_words)
    model.eval()
    criterion = model.loss
    accuracies, losses = np.array([]), np.array([])
    for batch in dataloader:
        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        files = batch['file']
        output = model(frames)
        loss = criterion(output, labels.squeeze(1))

        acc = accuracy(output, labels, files, words)
        losses = np.append(losses, loss.item())
        accuracies = np.append(accuracies, acc)

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    print("loss: %.3f, acc %.5f" % (avg_loss, avg_acc))


def accuracy(output, labels, files, words):
    sums = torch.sum(output, dim=1)
    _, predicted = sums.max(dim=1)
    labels = labels.squeeze(dim=1)
    correct = 0
    for i in range(len(labels)):
        if predicted[i] == labels[i]:
            correct += 1
        else:
            print("Expected: %s, Actual: %s, File: %s" % (words[labels[i]], words[predicted[i]], files[i]))
    return correct / output.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--words", type=int, default=10)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--set", type=str, default='val')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    query = None # '(yaw > 30)'
    dataset = HDF5Dataset(path=args.hdf5, table=args.set, query=query, columns=['frames', 'label', 'file'])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
    model = Model(num_classes=args.words, resnet_layers=args.resnet, resnet_pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    evaluate(model, dataloader, args.words, args.data)
