import glob
import os
from string import ascii_lowercase

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.preprocess.face import FacePredictor
from src.data.preprocess.video import load_mouth_images, load_video


class GridDataset(Dataset):
    def __init__(self, path, mode="train"):
        self.path = path
        if mode == "train":
            self.speakers = (0, 24)
        elif mode == "val":
            self.speakers = (24, 29)
        else:
            self.speakers = (29, 34)
        self.file_list = self.build_file_list()
        self.character_map = self.build_character_map()
        self.predictor = FacePredictor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file, align = self.file_list[idx]
        # frames = load_mouth_images(self.predictor, file)
        frames = load_video(file)
        chars = self.load_characters(align)

        width = 112 # 60
        height = 112 # 40
        max_frames = 75
        frames = torch.zeros(1, max_frames, width, height)
        for i, frame in enumerate(frames):
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((112, 112)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.4161, ], [0.1688, ]),
            ])(frame)
            frames[0][i] = result

        return {
            "frames": frames,
            "chars": chars,
            "input_lengths": max_frames,
            "output_lengths": 1,
        }

    def build_character_map(self):
        vocab_map = {' ': 0}
        for i, char in enumerate(ascii_lowercase):
            vocab_map[char] = i + 1
        return vocab_map

    def build_file_list(self):
        pattern = self.path + "/videos/**/*.mpg"
        all_files = glob.glob(pattern)
        files = []
        for file in all_files:
            speaker = int(file.split("/")[-2][1:])
            video_name = file.split("/")[-1][:-4]
            if speaker >= self.speakers[0] and speaker < self.speakers[1]:
                align = os.path.join(self.path, 'aligns', "s" + str(speaker), video_name + '.align')
                sample = (file, align)
                files.append(sample)

        return files

    def load_characters(self, align):
        file = open(align, "r")
        lines = file.readlines()
        chars = []
        for line in lines:
            word = line.split(' ')[2].rstrip()
            if word == 'sil':
                continue
            for char in word:
                chars.append(self.character_map[char])
            chars.append(self.character_map[' '])

        chars = chars[:-1]
        file.close()
        return torch.LongTensor(chars)

    def load_video(self, file):
        cap = cv2.VideoCapture(file)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(image)

        return frames
