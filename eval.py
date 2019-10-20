import argparse

from pytorch_trainer.trainer import Trainer

from src.checkpoint import load_checkpoint
from src.models.lrw_model import LRWModel
from src.models.model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--words", type=int, default=10)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    query = None if args.query == None else [float(x) for x in args.query.strip().split(",")]
    assert query == None or len(query) == 2, "--query param not in format -20,20"
    args.pretrained = False if args.checkpoint != None else args.pretrained

    model = LRWModel(
        hparams=args,
        in_channels=1,
        augmentations=False,
        query=query,
    )
    trainer = Trainer(
        seed=args.seed,
        gpu_id=0,
    )
    load_checkpoint(args.checkpoint, model)
    results = trainer.test(model)
    print(results)
