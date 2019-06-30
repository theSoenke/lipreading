import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tables import Float32Col, Int32Col, IsDescription, open_file


class Video(IsDescription):
    label = Int32Col()
    frames = Float32Col(shape=(29, 112, 112))

class HDF5Dataset(Dataset):
    def __init__(self, path, mode='train'):
        self.path = path
        h5file = open_file(path, mode="r", libver='latest', swmr=True)
        self.table = h5file.root.lrw[mode]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table[idx]
        label = row['label']
        frames = torch.from_numpy(row['frames']).unsqueeze(0)
        sample = {'input': frames, 'label': torch.LongTensor([label])}
        return sample


def preprocess_hdf5(dataset, output_path, table):
    file = open_file(output_path, mode="a")
    table = file.create_table("/lrw", table, Video, createparents=True)
    sample_row = table.row

    for sample in tqdm(dataset):
        sample_row['frames'] = sample['input'].numpy()
        sample_row['label'] = sample['label'].numpy()
        sample_row.append()
    table.flush()
    file.close()
