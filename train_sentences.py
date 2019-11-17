import argparse

import psutil
import torch
from pytorch_trainer import (EarlyStopping, ModelCheckpoint, Trainer,
                             WandbLogger)

from src.checkpoint import load_checkpoint
from src.models.attention_net import AttentionLRNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/datasets/lrs2")
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/lrs2')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--pretrain", default=False, action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers
    args.pretrained = False if args.checkpoint != None else args.pretrained
    model = AttentionLRNet(
        hparams=args,
        in_channels=1,
        pretrain=args.pretrain,
    )
    logger = WandbLogger(
        project='lrs2_wlsnet',
        model=model,
    )
    model.logger = logger
    trainer = Trainer(
        seed=args.seed,
        logger=logger,
        gpu_id=0,
        num_max_epochs=args.epochs,
        use_amp=False,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    logger.log('parameters', trainable_params)
    logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(
        directory=args.checkpoint_dir,
        save_best_only=True,
        monitor='val_cer',
        mode='min',
        prefix="lrs2_wlsnet",
    )

    trainer.fit(model)
    logger.save_file(checkpoint_callback.last_checkpoint_path)
