import math
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

from src.data.transforms import Crop


class LRS2Dataset(Dataset):
    def __init__(self, path, mode, in_channels=1, max_timesteps=100, skip_long_samples=True, max_text_len=200, pretrain_words=0, pretrain=False):
        assert mode in ['train', 'val', 'test']
        self.max_timesteps = max_timesteps
        self.pretrain = pretrain
        self.in_channels = in_channels
        self.max_timesteps = max_timesteps
        self.skip_long_samples = skip_long_samples
        self.max_text_len = max_text_len
        self.pretrain_words = pretrain_words
        self.file_paths, self.file_names, self.crops = self.build_file_list(path, mode)
        self.char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                          'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8',  '9', '0', '<sos>', '<eos>', '<pad>', '\'', ' ']
        self.int2char = dict(enumerate(self.char_list))
        self.char2int = {char: index for index, char in self.int2char.items()}

    def build_file_list(self, directory, mode):
        file_list, paths = [], []
        crops = {}
        skipped_samples = 0

        if self.pretrain:
            path = f"data/preprocess/lrs2/pretrain_crop.txt"
        else:
            path = f"data/preprocess/lrs2/{mode}_crop.txt"

        file = open(path, "r")
        content = file.read()
        for i, line in enumerate(content.splitlines()):
            split = line.split(":")
            file = split[0]
            crop_str = split[1]
            crops[file] = crop_str

        if self.pretrain:
            file = open(f"{directory}/pretrain.txt", "r")
            content = file.read()
            for file in content.splitlines():
                if file in crops:
                    file_list.append(file)
                    paths.append(f"{directory}/mvlrs_v1/pretrain/{file}")

            split = int(len(paths) * 0.95)
            if mode == 'train':
                paths = paths[:split]
                file_list = file_list[:split]
            elif mode == 'val':
                paths = paths[split:]
                file_list = file_list[split:]

        else:
            file = open(f"{directory}/{mode}.txt", "r")
            content = file.read()
            for file in content.splitlines():
                file = file.split(" ")[0]
                if file not in crops:
                    continue

                if self.skip_long_samples:
                    if crops[file].count("|") < self.max_timesteps:
                        file_list.append(file)
                        paths.append(f"{directory}/mvlrs_v1/main/{file}")
                    else:
                        skipped_samples += 1
                else:
                    file_list.append(file)
                    paths.append(f"{directory}/mvlrs_v1/main/{file}")

        if self.skip_long_samples:
            print(f"Skipped {skipped_samples} too long samples")
        return paths, file_list, crops

    def build_tensor(self, frames, crops):
        crops = [crop.split(";") for crop in crops]
        for i, crop_frame in enumerate(crops):
            crop = [float(crop) for crop in crop_frame]
            crops[i] = crop

        temporalVolume = torch.zeros(self.max_timesteps, self.in_channels, 64, 96)
        for i, frame in enumerate(frames):
            if self.in_channels == 1:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    Crop(crops[i]),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4161, ], [0.1688, ]),
                ])
            elif self.in_channels == 3:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    Crop(crops[i]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            temporalVolume[i] = transform(frame)

        temporalVolume = temporalVolume.transpose(1, 0)  # (C, D, H, W)
        return temporalVolume

    def __len__(self):
        return len(self.file_paths)

    def get_pretrain_words(self, content):
        lines = content.splitlines()[4:]
        words = []
        for line in lines:
            word, start, stop, _ = line.split(" ")
            start, stop = float(start), float(stop)
            words.append([word, start, stop])

        num_words = min(random.randint(self.pretrain_words - 1, self.pretrain_words), len(words))
        word_start = random.randint(0, len(words) - num_words)
        word_end = word_start + num_words

        sample_start = 0
        sample_end = 0
        content = ""
        for word in words[word_start:word_end]:
            word, start, end = word
            if sample_start == 0:
                sample_start = start
            if end > sample_end:
                sample_end = end
            content = content + " " + word

        return content, sample_start, sample_end

    def __getitem__(self, idx):
        file = self.file_names[idx]
        file_path = self.file_paths[idx]
        content = open(file_path + ".txt", "r").read()

        frame_crops = self.crops[file].split("|")
        start_sec = 0
        stop_sec = None
        if self.pretrain:
            content, start_sec, stop_sec = self.get_pretrain_words(content)
        else:
            content = content.splitlines()[0][7:]
            crop = frame_crops

        video, _, info = torchvision.io.read_video(file_path + ".mp4", start_pts=start_sec, end_pts=stop_sec, pts_unit='sec')  # T, H, W, C
        video = video.permute(0, 3, 1, 2)  # T C H W
        num_frames = video.size(0)

        if num_frames > self.max_timesteps:
            print(f"Cutting frames off. Requires {len(video)} frames: {file}")
            video = video[:self.max_timesteps]
            num_frames = video.size(0)

        if self.pretrain:
            fps = info['video_fps']
            start_frame = int(start_sec * fps)
            crop = frame_crops[start_frame:start_frame + num_frames]

        crop = crop[:self.max_timesteps]
        assert num_frames <= self.max_timesteps, f"Video too large with {num_frames} frames: {file_path}"
        content = content.strip().upper()

        assert len(crop) == num_frames
        assert len(content) >= 1
        frames = self.build_tensor(video, crop)

        encoded = self.encode(content)
        return frames, num_frames, encoded

    def encode(self, content):
        encoded = [self.char2int[c] for c in content] + [self.char2int['<eos>']]
        if len(encoded) > self.max_text_len:
            print(f"Max output length too short. Required {len(encoded)}")
            encoded = encoded[:self.max_text_len]
        encoded += [self.char2int['<pad>'] for _ in range(self.max_text_len - len(encoded))]
        return torch.Tensor(encoded)
