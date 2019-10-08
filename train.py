import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import wandb
from src.models.lrw_model import LRWModel
from src.wandb_logger import WandbLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints')
    parser.add_argument("--checkpoint", type=str)
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

    checkpoint_callback = ModelCheckpoint(
        filepath=args.checkpoint_dir,
        save_best_only=True,
        verbose=True,
        monitor='val_acc',
        mode='min',
        prefix=''
    )
    pretrained = False if args.checkpoint != None else args.pretrained
    model = LRWModel(
        hparams=args,
        resnet_layers=args.resnet,
        resnet_pretrained=pretrained,
    )
    logger = WandbLogger(
        project='lipreading',
        model=model,
    )
    trainer = Trainer(
        logger=logger,
        gpus=[1],
        default_save_path=args.tensorboard_logdir,
        checkpoint_callback=checkpoint_callback
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # if args.checkpoint != None:
    #     load_checkpoint(args.checkpoint, model, optimizer=None)

    # wandb.config.parameters = trainable_params
    # wandb.config.query = query
    trainer.fit(model)
    # wandb.save(checkpoint_path)
