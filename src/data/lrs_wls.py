import os
from glob import glob

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.charset import get_charSet, init_charSet


class LRS2Dataset(Dataset):
    def __init__(self, path, mode, max_timesteps=100, txtMaxLen=100):
        self.file_paths = self.build_file_list(path, mode)
        self.max_timesteps = max_timesteps
        self.txtMaxLen = txtMaxLen

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video, length = self.videoProcess(self.file_paths[idx])
        return video, length, txtProcess(self.file_paths[idx] + ".txt", self.txtMaxLen)

    def build_file_list(self, directory, mode):
        paths = []

        file = open(f"{directory}/{mode}.txt", "r")
        content = file.read()
        for file in content.splitlines():
            file = file.split(" ")[0]
            paths.append(f"{directory}/mvlrs_v1/main/{file}")

        return paths

    def build_tensor(self, frames):
        temporalVolume = torch.zeros(self.max_timesteps, 1, 120, 120)
        for i, frame in enumerate(frames):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((120, 120)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.4161, ], [0.1688, ]),
            ])
            temporalVolume[i] = transform(frame)

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, D, H, W)
        return temporalVolume

    def videoProcess(self, path):
        video, _, info = torchvision.io.read_video(path + ".mp4", pts_unit='sec')  # T, H, W, C
        video = video.permute(0, 3, 1, 2)[:self.max_timesteps]  # T C H W

        if len(video) > self.max_timesteps:
            print(f"Cutting off frames: {path}")
            video = video[:self.max_timesteps]
        frames = self.build_tensor(video)

        return frames, frames.size(0)


def txtProcess(dir, txtMaxLen):
    encoded = []
    with open(dir) as f:
        encoded = [get_charSet().get_index_of(i) for i in f.readline().split(':')[1].strip()] + [get_charSet().get_index_of('<eos>')]
        if len(encoded) > txtMaxLen:
            print(f'too short txt max length. Required: {len(encoded)}')
            encoded = encoded[:txtMaxLen]
        else:
            encoded += [get_charSet().get_index_of('<pad>') for _ in range(txtMaxLen - len(encoded))]
    return torch.Tensor(encoded)
