import os

import imageio
import psutil
import torch
import torchvision.transforms.functional as F
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.preprocess.pose_fa import HeadPose


class OuluVS2Dataset(Dataset):
    def __init__(self, directory):
        self.file_list = self.build_file_list(directory)
        self.head_pose = HeadPose(use_cuda=True)

    def build_file_list(self, directory):
        videos = []
        files = os.listdir(directory)
        for file in files:
            if file.endswith("mp4"):
                filepath = directory + "/{}".format(file)
                videos.append(filepath)
        return videos

    def load_video(self, file):
        video = imageio.get_reader(file,  'ffmpeg')
        image = video.get_data(0)
        try:
            angles = self.head_pose.predict(image)
            if angles == None:
                print("File: %s, Error: Could not detect pose" % (file))
                yaw = None
            else:
                yaw = angles['yaw']
        except Exception as e:
            print("File: %s, Error: %s" % (file, e))
            yaw = None

        return yaw

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        yaw = self.load_video(file)
        split = file.split("/")[-1][:-4].split("_")
        speaker, view, utterance = [int(x[1:]) for x in split]
        if yaw == None:
            yaw = 500
        view = [0, 30, 45, 60, 90][view-1]
        sample = {
            'yaw': yaw,
            "view": view,
            "speaker": speaker,
            "file": file,
            "utterance": utterance,
        }
        return sample


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
    dataset = OuluVS2Dataset(directory=path)
    preprocess_hdf5(
        dataset=dataset,
        output_path=output_path,
        table='train',
        workers=workers,
    )


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
