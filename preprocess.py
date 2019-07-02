import argparse
import os

from src.data.hdf5 import preprocess_hdf5
from src.data.lrw import LRWDataset


def preprocess_lrw(path, output, num_classes):
    if os.path.exists(output) == False:
        os.makedirs(output)

    output_path = "%s/lrw_%d.h5" % (output, num_classes)
    if os.path.exists(output_path):
        os.remove(output_path)

    labels = None
    for mode in ['train', 'val', 'test']:
        print("Generating %s data" % mode)
        dataset = LRWDataset(directory=path, num_words=num_classes, mode=mode)
        if labels != None:
            assert labels == dataset.labels
        labels = dataset.labels
        preprocess_hdf5(
            dataset=dataset,
            output_path=output_path,
            table=mode
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--output', default='data/preprocessed')
    parser.add_argument('--words', type=int, default=10)
    args = parser.parse_args()

    preprocess_lrw(args.data, args.output, args.words)
