import argparse

import psutil
import torch
from pytorch_trainer import (EarlyStopping, ModelCheckpoint, Trainer,
                             WandbLogger)

from src.checkpoint import load_checkpoint
from src.models.lrs2_model import LRS2Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/datasets/lrs2")
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/lrs2')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        directory=args.checkpoint_dir,
        save_best_only=True,
        monitor='val_cer',
        mode='min',
        prefix=f"lrs2"
    )

    early_stop_callback = EarlyStopping(
        monitor='val_cer',
        min_delta=0.00,
        patience=10,
        mode='min'
    )

    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers
    args.pretrained = False if args.checkpoint != None else args.pretrained
    model = LRS2Model(
        hparams=args,
        in_channels=1,
        augmentations=False,
    )
    logger = WandbLogger(
        project='lrs2',
        model=model,
    )
    model.logger = logger
    trainer = Trainer(
        seed=args.seed,
        logger=logger,
        gpu_id=0,
        num_max_epochs=args.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    logger.log('parameters', trainable_params)
    logger.log_hyperparams(args)

    if args.checkpoint != None:
        load_checkpoint(args.checkpoint, model, optimizer=None)
        logs = trainer.validate(model)
        logger.log_metrics(logs)
        print(f"Initial validation: wer: {logs['val_wer']:.4f}, cer: {logs['val_cer']:.4f}")

    trainer.validate(model)
    trainer.fit(model)
    logger.save_file(checkpoint_callback.last_checkpoint_path)
