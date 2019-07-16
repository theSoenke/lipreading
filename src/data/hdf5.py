import numpy as np
import torch
from tables import open_file
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, path, table='train', columns=['frames', 'label']):
        self.path = path
        self.columns = columns
        h5file = open_file(path, mode="r")
        self.table = h5file.root[table]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table[idx]
        sample = {}
        for column in self.columns:
            value = row[column]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).unsqueeze(0)
            elif isinstance(value, np.int32):
                value = torch.LongTensor([value])
            else:
                value = str(value)
            sample[column] = value
        return sample
