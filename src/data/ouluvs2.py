import os
from string import ascii_lowercase

import cv2
import psutil
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class OuluVS2Dataset(Dataset):
    def __init__(self, directory, mode='train'):
        self.directory = directory
        self.file_list = self.build_file_list()
        self.character_map = self.build_character_map()

    def build_character_map(self):
        vocab_map = {' ': 0}
        for i, char in enumerate(ascii_lowercase):
            vocab_map[char] = i + 1
        return vocab_map

    def sentence_chars(self, sentence):
        mapping = []
        for char in sentence:
            mapping.append(self.character_map[char])
        return mapping

    def build_file_list(self):
        videos = []
        speakers = os.listdir(self.directory)
        for speaker in speakers:
            speaker_dir = os.path.join(self.directory, speaker)
            views = os.listdir(speaker_dir)
            for view in views:
                view_dir = os.path.join(speaker_dir, view)
                files = os.listdir(view_dir)
                for file in files:
                    videos.append(file)
        return videos

    def load_video(self, file):
        cap = cv2.VideoCapture(file)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(image)

        return self.build_tensor(frames)

    def build_tensor(self, frames):
        temporalVolume = torch.FloatTensor(1, 29, 112, 112)
        for i in range(len(frames)):
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.4161, ], [0.1688, ]),
            ])(frames[i])
            temporalVolume[0][i] = result

        return temporalVolume

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        frames, sentence = self.load_video(path)
        split = path.split("/")[-1][:-4].split("_")
        speaker, view, utterance = [int(x[1:]) for x in split]
        view = [0, 30, 45, 60, 90][view-1]
        sample = {
            'frames': frames,
            'length': len(frames),
            'chars': self.sentence_chars(sentence),
            "view": view,
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
    print("Saved preprocessed file: %s" % output_path)


def preprocess_hdf5(dataset, output_path, table, workers):
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video, expectedrows=len(dataset))
    row = table.row
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['frames'])):
                for column in batch:
                    row[column] = batch[column][i]
                row.append()
                progress.update(1)
    table.flush()
    file.close()
