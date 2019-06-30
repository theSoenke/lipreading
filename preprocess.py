import argparse

from src.data.hdf5 import preprocess_hdf5
from src.data.lrw import LRWDataset


def preprocess_lrw(path, num_classes):
    labels = None
    for mode in ['train', 'val', 'train']:
        print("Generating %s data" % mode)
        dataset = LRWDataset(directory=path, num_words=num_classes, mode=mode)
        if labels != None:
            assert labels == dataset.labels
        labels = dataset.labels
        preprocess_hdf5(
            dataset=dataset,
            output_path='data/preprocessed/lrw_%d.h5' % num_classes,
            group_name='lrw',
            table_name=mode
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()

    preprocess_lrw(args.data, 10)
