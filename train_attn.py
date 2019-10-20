import argparse

import psutil
import torch
from pytorch_trainer.early_stopping import EarlyStopping
from pytorch_trainer.model_checkpoint import ModelCheckpoint
from pytorch_trainer.trainer import Trainer
from pytorch_trainer.wandb_logger import WandbLogger

from src.models.expert_model import ExpertModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/lrw')
    parser.add_argument("--checkpoint_left", type=str, required=True)
    parser.add_argument("--checkpoint_center", type=str, required=True)
    parser.add_argument("--checkpoint_right", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--words", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        directory=args.checkpoint_dir,
        save_best_only=True,
        monitor='val_acc',
        mode='max',
        prefix=f"lrw_{args.words}"
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=10,
        mode='max'
    )

    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers
    args.pretrained = False
    model = ExpertModel(args, args.checkpoint_left, args.checkpoint_center, args.checkpoint_right)
    logger = WandbLogger(
        project='lipreading',
        model=model,
    )
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

    logs = trainer.validate(model)
    print(f"Initial expert val_acc: {logs['val_acc']:.4f}")
    trainer.fit(model)
