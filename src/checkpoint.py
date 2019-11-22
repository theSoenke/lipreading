import torch


def load_checkpoint(path, model, optimizer=None, strict=True):
    print("Loading checkpoint: %s" % path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'], strict=strict)
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def create_checkpoint(path, model, optimizer=None):
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    if optimizer != None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint_mismatch(path, model, verbose=False):
    model_dict = model.state_dict()
    checkpoint_dict = torch.load(path)['state_dict']

    pretrained_dict = {}
    for k, v in checkpoint_dict.items():
        if k not in model_dict:
            continue
        if v.shape != model_dict[k].shape:
            if log:
                print(f"Shape mismatch for {k}. Expected {model_dict[k].shape}, got {v.shape}")
            continue

        pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
