import torch


def load_checkpoint(path, model, optimizer=None):
    print("Loading checkpoint: %s" % path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def create_checkpoint(path, model, optimizer=None):
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    if optimizer != None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)
