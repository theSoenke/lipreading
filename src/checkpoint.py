import os

import torch


def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        print("Loading checkpoint: %s" % path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def create_checkpoint(model, optimizer, path):
    torch.save(
        {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
