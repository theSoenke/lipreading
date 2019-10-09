import argparse

import psutil

from src.checkpoint import load_checkpoint
from src.models.lipnet_model import LipNetModel
from src.trainer.early_stopping import EarlyStopping
from src.trainer.model_checkpoint import ModelCheckpoint
from src.trainer.trainer import Trainer
from src.trainer.wandb_logger import WandbLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/grid')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        directory=args.checkpoint_dir,
        save_best_only=True,
        monitor='val_acc',
        mode='max',
        prefix=f"grid"
    )
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=10,
        mode='max'
    )

    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers
    args.pretrained = False if args.checkpoint != None else args.pretrained
    model = LipNetModel(args)
    logger = WandbLogger(
        project='grid',
        model=model,
    )
    trainer = Trainer(
        logger=logger,
        gpu_id=0,
        num_max_epochs=args.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    logger.log('parameters', trainable_params)

    if args.checkpoint != None:
        load_checkpoint(args.checkpoint, model, optimizer=None)

    trainer.fit(model)
    logger.save_file(checkpoint_callback.last_checkpoint_path)
