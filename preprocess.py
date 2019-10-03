import argparse
import random

from src.data.lrw import preprocess as process_lrw
from src.data.ouluvs2 import preprocess as process_ouluvs2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('set')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='data/preprocessed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--words', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--augmentation', help='Augment data', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.set == "lrw":
        process_lrw(args.data, args.output, num_words=args.words, workers=args.workers, augmentation=args.augmentation)
    elif args.set == "ouluvs2":
        process_ouluvs2(args.data, args.output, workers=args.workers)
