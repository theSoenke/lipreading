import argparse
import os

import psutil

from src.data.lrw import extract_angles as lrw_extract_angles
from src.data.lrw import preprocess as process_lrw
from src.data.ouluvs2 import preprocess as process_ouluvs2
from src.preprocess.lrs2 import extract_angles as lrs2_extract_angles
from src.preprocess.lrs2 import \
    mouth_bounding_boxes as lrs2_mouth_bounding_boxes
from src.preprocess.lrs2 import \
    prepare_language_model as lrs2_prepare_language_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('set', type=str)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='data/preprocess')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--augmentation', help='Augment data', action='store_true')
    args = parser.parse_args()

    output_path = os.path.join(args.output, args.set)
    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers

    if args.set == "lrw":
        lrw_extract_angles(args.data, output_path=output_path, num_workers=args.workers, seed=args.seed)
        # process_lrw(args.data, args.output, num_words=args.words, workers=args.workers, augmentation=args.augmentation)
    elif args.set == "ouluvs2":
        process_ouluvs2(args.data, output_path, workers=args.workers)
    elif args.set == "lrs2":
        # lrs2_extract_angles(args.data, output_path=output_path, num_workers=args.workers)
        lrs2_mouth_bounding_boxes(args.data, output_path=output_path)
        # lrs2_prepare_language_model(args.data, output_path)
    else:
        raise Exception("Not a valid set name")
