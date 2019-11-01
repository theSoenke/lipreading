import os
import random
from string import ascii_lowercase

import psutil
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class LRS2Dataset(Dataset):
    def __init__(self, path, in_channels=1, mode="train", augmentations=False, estimate_pose=False):
        self.max_timesteps = 155
        self.in_channels = in_channels
        self.augmentation = augmentations if mode == 'train' else False
        self.file_paths, self.file_names = self.build_file_list(path, mode)
        self.estimate_pose = estimate_pose

        numbers = "".join([str(i) for i in range(10)])
        special_characters = " '"
        self.characters = ascii_lowercase + numbers + special_characters
        int2char = dict(enumerate(self.characters))
        self.char2int = {char: index for index, char in int2char.items()}

    def build_file_list(self, directory, mode):
        file = open(f"{directory}/{mode}.txt", "r")
        content = file.read()
        file_list = []
        paths = []
        for file in content.splitlines():
            file = file.split(" ")[0]
            file_list.append(file)
            paths.append(f"{directory}/mvlrs_v1/main/{file}")

        return paths, file_list

    def build_tensor(self, frames):
        temporalVolume = torch.zeros(self.max_timesteps, self.in_channels, 112, 112)
        if(self.augmentation):
            augmentations = transforms.Compose([])  # TODO
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

        for i in range(0, len(frames)):
            frame = frames[i].permute(2, 0, 1)  # (C, H, W)
            temporalVolume[i] = transform(frame)

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, D, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file = self.file_names[idx]
        video, _, _ = torchvision.io.read_video(self.file_paths[idx] + ".mp4")  # T, H, W, C
        frames = self.build_tensor(video)
        content = open(self.file_paths[idx] + ".txt", "r").read()
        sentence = content.splitlines()[0][7:].lower()
        encoded = []
        for i, char in enumerate(sentence):
            encoded.append(self.char2int[char])

        input_lengths = video.size(0)
        if self.estimate_pose:
            return frames, encoded, input_lengths, idx, file

        return frames, encoded, input_lengths, idx


def extract_angles(path, output_path, num_workers):
    from src.data.preprocess.pose_hopenet import HeadPose
    head_pose = HeadPose()

    os.makedirs(output_path, exist_ok=True)
    for mode in ['train', 'val', 'test']:
        dataset = LRS2Dataset(path=path, mode=mode, in_channels=3, estimate_pose=True)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers)
        lines = []
        with tqdm(total=len(dataset)) as progress:
            for batch in data_loader:
                frames, _, input_lengths,  _, files = batch
                for i, video in enumerate(frames):
                    video = video.transpose(1, 0)[:input_lengths[i]]  # T C H W
                    yaws = head_pose.predict(video)['yaw']
                    yaws = ";".join([f"{yaw:.2f}" for yaw in yaws.cpu().numpy()])
                    line = f"{files[i]};{yaws}"
                    lines.append(line)
                    progress.update(1)
        file = open(f"{output_path}/{mode}.txt", "w")
        file.write('\n'.join(lines))
        file.close()
