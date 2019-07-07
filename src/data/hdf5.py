import torch
from tables import open_file
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, path, table='train'):
        self.path = path
        h5file = open_file(path, mode="r")
        self.table = h5file.root[table]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table[idx]
        label = row['label']
        frames = torch.from_numpy(row['frames']).unsqueeze(0)
        sample = {'frames': frames, 'label': torch.LongTensor([label])}
        return sample
