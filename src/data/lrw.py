import os
import random

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.transforms import StatefulRandomHorizontalFlip


def build_word_list(directory, num_words, seed):
    random.seed(seed)
    words = os.listdir(directory)
    words.sort()
    random.shuffle(words)
    words = words[:num_words]
    return words


class LRWDataset(Dataset):
    def __init__(self, path, num_words=500, in_channels=1, mode="train", augmentations=False, estimate_pose=False, seed=42, query=None):
        self.seed = seed
        self.num_words = num_words
        self.in_channels = in_channels
        self.query = query
        self.augmentation = augmentations if mode == 'train' else False
        self.poses = None
        if estimate_pose == False:
            self.poses = self.head_poses(mode, query)
        self.video_paths, self.files, self.labels, self.words = self.build_file_list(path, mode)
        self.estimate_pose = estimate_pose

    def head_poses(self, mode, query):
        poses = {}
        yaw_file = open(f"data/preprocess/lrw/{mode}.txt", "r")
        content = yaw_file.read()
        for line in content.splitlines():
            file, yaw = line.split(",")
            yaw = float(yaw)
            if query == None or (query[0] <= yaw and query[1] > yaw):
                poses[file] = yaw
        return poses

    def build_file_list(self, directory, mode):
        words = build_word_list(directory, self.num_words, seed=self.seed)
        print(words)
        paths = []
        file_list = []
        labels = []
        for i, word in enumerate(words):
            dirpath = directory + "/{}/{}".format(word, mode)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    if self.poses != None and file not in self.poses:
                        continue
                    path = dirpath + "/{}".format(file)
                    file_list.append(file)
                    paths.append(path)
                    labels.append(i)

        return paths, file_list, labels, words

    def build_tensor(self, frames):
        temporalVolume = torch.FloatTensor(29, self.in_channels, 112, 112)
        if(self.augmentation):
            augmentations = transforms.Compose([
                StatefulRandomHorizontalFlip(0.5),
            ])
        else:
            augmentations = transforms.Compose([])

        if self.in_channels == 1:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((112, 112)),
                augmentations,
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.4161, ], [0.1688, ]),
            ])
        elif self.in_channels == 3:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((112, 112)),
                augmentations,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        for i in range(0, 29):
            frame = frames[i].permute(2, 0, 1)  # (C, H, W)
            temporalVolume[i] = transform(frame)

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, D, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file = self.files[idx]
        video, _, _ = torchvision.io.read_video(self.video_paths[idx])  # (Tensor[T, H, W, C])
        if self.estimate_pose:
            angle_frame = video[14].permute(2, 0, 1)
        else:
            angle_frame = 0
        frames = self.build_tensor(video)
        if self.estimate_pose:
            yaw = 0
        else:
            yaw = self.poses[file]

        sample = {
            'frames': frames,
            'label': torch.LongTensor([label]),
            'word': self.words[label],
            'file': self.files[idx],
            'yaw': torch.FloatTensor([yaw]),
            'angle_frame': angle_frame,
        }
        return sample
