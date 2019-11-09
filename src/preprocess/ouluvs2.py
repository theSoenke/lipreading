import glob
import os

import psutil
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.ouluvs2 import OuluVS2Dataset
from src.data.transforms import StatefulRandomHorizontalFlip


class Video(IsDescription):
    yaw = Float32Col()
    view = Float32Col()
    speaker = Int32Col()
    utterance = Int32Col()
    file = StringCol(32)


def preprocess(path, output, workers=None):
    workers = psutil.cpu_count() if workers == None else workers
    if os.path.exists(output) == False:
        os.makedirs(output)

    output_path = "%s/ouluvs2.h5" % (output)
    if os.path.exists(output_path):
        os.remove(output_path)

    labels = None
    dataset = OuluVS2Dataset(path=path)
    preprocess_hdf5(
        dataset=dataset,
        output_path=output_path,
        table='train',
        workers=workers,
    )
    print("Saved preprocessed file: %s" % output_path)


def preprocess_hdf5(dataset, output_path, table, workers):
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video, expectedrows=len(dataset))
    row = table.row
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['yaw'])):
                for column in batch:
                    row[column] = batch[column][i]
                row.append()
                progress.update(1)
    table.flush()
    file.close()
