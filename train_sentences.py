import argparse

import psutil
import torch
from pytorch_trainer import (EarlyStopping, ModelCheckpoint, Trainer,
                             WandbLogger)

from src.checkpoint import load_checkpoint
from src.models.lrs2_model import LRS2Model
from src.models.lrs2_resnet_attn import LRS2ResnetAttn
from src.models.wlsnet import WLSNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/datasets/lrs2")
    parser.add_argument('--model', default="resnet")
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/lrs2')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--pretrain", default=False, action='store_true')
    parser.add_argument("--use_amp", default=False, action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers
    args.pretrained = False if args.checkpoint != None else args.pretrained

    if args.model == 'resnet':
        model = LRS2ResnetAttn(
            hparams=args,
            in_channels=1,
            pretrain=args.pretrain,
        )
    elif args.model == 'wlsnet':
        model = WLSNet(
            hparams=args,
            in_channels=1,
            pretrain=args.pretrain,
        )
    else:
        model = LRS2Model(
            hparams=args,
            in_channels=1,
            augmentations=False,
            pretrain=args.pretrain,
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
        use_amp=args.use_amp,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    logger.log('parameters', trainable_params)
    logger.log_hyperparams(args)

    if args.checkpoint is not None:
        load_checkpoint_mismatch(args.checkpoint, model)
        logs = trainer.validate(model)
        logger.log_metrics(logs)
        print(f"Initial validation: wer: {logs['val_wer']:.4f}, cer: {logs['val_cer']:.4f}")

    if args.pretrain:
        print("Pretraining model")

        # curriculum with max_sequence_length, max_text_len, number_of_words, epochs
        curriculum = [
            [64, 32, 2, 15],
            [96, 40, 3, 15],
            [120, 48, 4, 10],
            [132, 56, 6, 5],
            [148, 64, 8, 5],
        ]

        for part in curriculum:
            checkpoint_callback = ModelCheckpoint(
                directory=args.checkpoint_dir,
                period=part[3],
                prefix=f"lrs2_pretrain_{part[2]}",
            )

            trainer.checkpoint_callback = checkpoint_callback
            model.max_timesteps = part[0]
            model.max_text_len = part[1]
            model.pretrain_words = part[2]
            trainer.num_max_epochs = part[3]
            trainer.val_percent = 0.0
            trainer.fit(model)
            logger.save_file(checkpoint_callback.last_checkpoint_path)

        trainer.validate(model)
        print("Pretraining finished")

    if args.checkpoint != None:
        load_checkpoint(args.checkpoint, model, optimizer=None)
        logs = trainer.validate(model)
        logger.log_metrics(logs)
        print(f"Initial validation: wer: {logs['val_wer']:.4f}, cer: {logs['val_cer']:.4f}")

    checkpoint_callback = ModelCheckpoint(
        directory=args.checkpoint_dir,
        save_best_only=True,
        monitor='val_cer',
        mode='min',
        prefix="lrs2",
    )

    trainer.checkpoint_callback = checkpoint_callback
    model.pretrain = False
    model.max_timesteps = 100
    model.max_text_len = 100
    trainer.num_max_epochs = args.epochs
    trainer.fit(model)

    logger.save_file(checkpoint_callback.last_checkpoint_path)
