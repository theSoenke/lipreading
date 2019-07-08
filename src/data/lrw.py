import os
import random

import imageio
import psutil
import torch
import torchvision.transforms.functional as F
from tables import Float32Col, Int32Col, IsDescription, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.preprocess.head_pose import HeadPose


class LRWDataset(Dataset):
    def __init__(self, directory, num_words=500, mode="train"):
        self.num_words = num_words
        self.file_list, self.labels = self.build_file_list(directory, mode)
        self.head_pose = HeadPose()

    def build_file_list(self, directory, mode):
        random.seed(42)
        labels = os.listdir(directory)
        labels.sort()
        random.shuffle(labels)
        labels = labels[:self.num_words]
        print(labels)
        videos = []

        for i, label in enumerate(labels):
            dirpath = directory + "/{}/{}".format(label, mode)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    filepath = dirpath + "/{}".format(file)
                    video = (i, filepath)
                    videos.append(video)

        return videos, labels

    def load_video(self, file):
        video = imageio.get_reader(file,  'ffmpeg')
        frames = []
        for i in range(0, 29):
            image = video.get_data(i)
            if i == 14:
                try:
                    angles = self.head_pose.predict(image)
                    yaw = -angles[1, 0]
                except:
                    yaw = None
            image = F.to_tensor(image)
            frames.append(image)

        return self.build_tensor(frames), yaw

    def build_tensor(self, frames):
        temporalVolume = torch.FloatTensor(1, 29, 112, 112)
        croptransform = transforms.CenterCrop((112, 112))

        for i in range(0, 29):
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((122, 122)),
                croptransform,
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.4161, ], [0.1688, ]),
            ])(frames[i])

            temporalVolume[0][i] = result

        return temporalVolume

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        label, filename = self.file_list[idx]
        frames, yaw = self.load_video(filename)
        if yaw == None:
            yaw = 1000
            # print("No face found: %s" % filename)
        sample = {'frames': frames, 'label': torch.LongTensor([label]), 'yaw': yaw}
        return sample


class Video(IsDescription):
    label = Int32Col()
    frames = Float32Col(shape=(29, 112, 112))
    yaw = Float32Col()


def preprocess(path, output, num_words):
    if os.path.exists(output) == False:
        os.makedirs(output)

    output_path = "%s/lrw_%d.h5" % (output, num_words)
    if os.path.exists(output_path):
        os.remove(output_path)

    labels = None
    for mode in ['train', 'val', 'test']:
        print("Generating %s data" % mode)
        dataset = LRWDataset(directory=path, num_words=num_words, mode=mode)
        if labels != None:
            assert labels == dataset.labels
        labels = dataset.labels
        preprocess_hdf5(
            dataset=dataset,
            output_path=output_path,
            table=mode
        )


def preprocess_hdf5(dataset, output_path, table):
    workers = psutil.cpu_count()
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video)
    row = table.row
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['yaw'])):
                for column in batch:
                    row[column] = batch[column][i].numpy()
                row.append()
                progress.update(1)
    table.flush()
    file.close()
