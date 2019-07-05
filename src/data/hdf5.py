import psutil
import os

import psutil
import torch
from tables import Float32Col, Int32Col, IsDescription, open_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Video(IsDescription):
    label = Int32Col()
    frames = Float32Col(shape=(29, 112, 112))
    yaw = Float32Col()


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
        sample = {'input': frames, 'label': torch.LongTensor([label])}
        return sample

def preprocess_hdf5(dataset, output_path, table):
    workers = psutil.cpu_count()
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video)
    sample_row = table.row
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['label'])):
                sample_row['frames'] = batch['input'][i].numpy()
                sample_row['label'] = batch['label'][i].numpy()
                sample_row['yaw'] = batch['yaw'][i].numpy()
                sample_row.append()
                progress.update(1)
    table.flush()
    file.close()
