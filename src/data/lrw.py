import os
import random

import cv2
import psutil
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.preprocess.pose_hopenet import HeadPose


def build_word_list(directory, num_words):
    random.seed(42)
    words = os.listdir(directory)
    words.sort()
    random.shuffle(words)
    words = words[:num_words]
    return words


class LRWDataset(Dataset):
    def __init__(self, directory, num_words=500, mode="train"):
        self.num_words = num_words
        self.file_list, self.words = self.build_file_list(directory, mode)
        self.head_pose = HeadPose()

    def build_file_list(self, directory, mode):
        words = build_word_list(directory, self.num_words)
        print(words)
        videos = []
        for i, word in enumerate(words):
            dirpath = directory + "/{}/{}".format(word, mode)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    path = dirpath + "/{}".format(file)
                    video = (i, path, file)
                    videos.append(video)

        return videos, words

    def load_video(self, file):
        cap = cv2.VideoCapture(file)
        frames = []
        for i in range(0, 29):
            _, frame = cap.read()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if i == 14:
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
        label, path, filename = self.file_list[idx]
        frames, yaw = self.load_video(path)
        if yaw == None:
            yaw = 500
        sample = {
            'frames': frames,
            'label': torch.LongTensor([label]),
            'yaw': yaw,
            'file': filename,
            'word': self.words[label],
        }
        return sample


class Video(IsDescription):
    label = Int32Col()
    frames = Float32Col(shape=(29, 112, 112))
    yaw = Float32Col()
    file = StringCol(32)
    word = StringCol(32)


def preprocess(path, output, num_words, workers=None):
    workers = psutil.cpu_count() if workers == None else workers
    if os.path.exists(output) == False:
        os.makedirs(output)

    output_path = "%s/lrw_%d.h5" % (output, num_words)
    if os.path.exists(output_path):
        os.remove(output_path)

    words = None
    for mode in ['train', 'val', 'test']:
        print("Generating %s data" % mode)
        dataset = LRWDataset(directory=path, num_words=num_words, mode=mode)
        if words != None:
            assert words == dataset.words
        words = dataset.words
        preprocess_hdf5(
            dataset=dataset,
            output_path=output_path,
            table=mode,
            workers=workers,
        )


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
