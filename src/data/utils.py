import torch
from torch.utils.data.dataloader import default_collate


def ctc_collate(batch):
    xs, ys, lens, indices = zip(*batch)
    max_len = max(lens)
    x = default_collate(xs)
    x.narrow(2, 0, max_len)
    y = []
    for sub in ys:
        y += sub
    y = torch.IntTensor(y)
    lengths = torch.IntTensor(lens)
    y_lengths = torch.IntTensor([len(label) for label in ys])
    ids = default_collate(indices)

    return x, y, lengths, y_lengths, ids
