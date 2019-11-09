import os
import random

import psutil
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.lrw import LRWDataset


def extract_angles(path, output_path, num_workers, seed):
    from src.preprocess.head_pose.hopenet import HeadPose
    head_pose = HeadPose()

    words = None
    for mode in ['train', 'val', 'test']:
        dataset = LRWDataset(path=path, num_words=500, mode=mode, estimate_pose=True, seed=seed)
        if words != None:
            assert words == dataset.words
        words = dataset.words
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
        lines = ""
        with tqdm(total=len(dataset)) as progress:
            for batch in data_loader:
                frames = batch['angle_frame']
                files = batch['file']
                yaws = head_pose.predict(frames)['yaw']
                for i in range(len(batch['frames'])):
                    line = f"{files[i]},{yaws[i].item():.2f}\n"
                    lines += line
                    progress.update(1)
        file = open(f"{output_path}/{mode}.txt", "w")
        file.write(lines)
        file.close()


class Video(IsDescription):
    label = Int32Col()
    frames = Float32Col(shape=(29, 112, 112))
    yaw = Float32Col()
    file = StringCol(32)
    word = StringCol(32)


def preprocess(path, output, num_words, augmentation=False, workers=None):
    workers = psutil.cpu_count() if workers == None else workers
    if os.path.exists(output) == False:
        os.makedirs(output)

    if augmentation:
        output_path = "%s/lrw_aug_%d.h5" % (output, num_words)
    else:
        output_path = "%s/lrw_%d.h5" % (output, num_words)
    if os.path.exists(output_path):
        os.remove(output_path)

    words = None
    for mode in ['train', 'val', 'test']:
        print("Generating %s data" % mode)
        dataset = LRWDataset(path=path, num_words=num_words, mode=mode, augmentations=augmentation, estimate_pose=True)
        if words != None:
            assert words == dataset.words
        words = dataset.words
        preprocess_hdf5(
            dataset=dataset,
            output_path=output_path,
            table=mode,
            workers=workers,
        )
    print("Saved preprocessed file: %s" % output_path)


def preprocess_hdf5(dataset, output_path, table, workers=0):
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video, expectedrows=len(dataset))
    row = table.row
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['yaw'])):
                for column in batch:
                    value = batch[column][i]
                    if isinstance(value, str):
                        row[column] = batch[column][i]
                    else:
                        row[column] = batch[column][i].numpy()
                row.append()
                progress.update(1)
    table.flush()
    file.close()
