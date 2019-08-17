import glob
import math
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.transforms import StatefulRandomHorizontalFlip


def round(x):
    return math.floor(x + 0.5)


class GRIDDataset(Dataset):
    def __init__(self, path, mode='train', augmentation=False):
        self.path = path
        self.mode = mode
        self.augmentation = augmentation
        self.max_timesteps = 75
        self.speakers = {
            'train': (0, 24),
            'val': (24, 29),
            'test': (29, 34),
        }[mode]
        self.file_list = self.build_file_list()
        self.dataset = []

        self.preprocess()
        print(f'{mode}: videos = {self.num_videos}, samples = {len(self.dataset)}, vocab_size = {len(self.vocab)}')
        print(f'vocab = {"|".join(self.vocab)}')

    def preprocess(self):
        vocab_unordered = {}
        self.num_videos = 0
        for video, align in self.file_list:
            speaker = int(video.split("/")[-2][1:])
            video = video.split("/")[-1][:-4]

            self.num_videos += 1
            sample = {'speaker': speaker, 'video': video, 'words': [], 'time_start': [], 'time_end': []}
            for line in open(align, 'r').read().splitlines():
                token = line.split(' ')
                if token[2] != 'sil' and token[2] != 'sp':
                    sample['words'].append(token[2])
                    sample['time_start'].append(int(token[0]))
                    sample['time_end'].append(int(token[1]))
                    for char in token[2]:
                        vocab_unordered[char] = True
            for start in range(1, 7):
                sample_i = sample.copy()
                sample_i['mode'], sample_i['word_start'] = 1, start
                sample_i['word_end'] = start + sample_i['mode'] - 1
                frame_start = max(round(1 / 1000 * sample['time_start'][sample_i['word_start'] - 1]), 1)
                frame_end = min(round(1 / 1000 * sample['time_end'][sample_i['word_end'] - 1]), 75)
                if frame_end - frame_start + 1 >= 3:
                    self.dataset.append(sample_i)

            sample_i = sample.copy()
            sample_i['mode'] = 7
            self.dataset.append(sample_i)

        self.vocab = [' ']
        for char in vocab_unordered:
            self.vocab.append(char)
        self.vocab.sort()
        self.vocab_mapping = {' ': 0}
        for i, char in enumerate(self.vocab):
            self.vocab_mapping[char] = i + 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = torch.zeros(3, self.max_timesteps, 40, 60)
        data = self.dataset[idx]
        frames, y = self.load_sample(data)
        x[:, : frames.size(1), :, :] = frames
        length = frames.size(1)

        return x, y, length, idx

    def load_sample(self, data):
        mode = data['mode']
        if mode < 7:
            word_start = data['word_start'] or random.randint(1, len(data['words']) - mode + 1)
            word_end = word_start + mode - 1

        min_frame, max_frame = 1, 75
        frame_start, frame_end = -1, -1

        if mode == 7:
            frame_start, frame_end = min_frame, max_frame
            sub = ' '.join(data['words'])
        else:
            words = []
            for w_i in range(word_start, word_end + 1):
                words.append(data['words'][w_i - 1])
            sub = ' '.join(words)

            frame_start = max(round(1 / 1000 * data['time_start'][word_start - 1]), 1)
            frame_end = min(round(1 / 1000 * data['time_end'][word_end - 1]), 75)

            if frame_end - frame_start + 1 <= 2:
                frame_start, frame_end = min_frame, max_frame
                sub = ' '.join(data['words'])

        y = []
        for char in sub:
            y.append(self.vocab_mapping[char])

        video_path = os.path.join(self.path, 'mouths', 's' + str(data['speaker']), data['video'])
        x = torch.FloatTensor(3, frame_end - frame_start + 1, 40, 60)
        if(self.augmentation):
            augmentations = transforms.Compose([
                StatefulRandomHorizontalFlip(0.5),
            ])
        else:
            augmentations = transforms.Compose([])

        transform = transforms.Compose([
            augmentations,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7136, 0.4906, 0.3283],
                                 std=[0.113855171, 0.107828568, 0.0917060521])
        ])
        frame_count = 0
        for frame_num in range(frame_start, frame_end + 1):
            file = '{}/mouth_{:03d}.png'.format(video_path, frame_num - 1)
            img = Image.open(file).convert('RGB')
            img = transform(img)
            x[:, frame_count, :, :] = img
            frame_count += 1

        return x, y

    def build_file_list(self):
        pattern = self.path + "/videos/**/*.mpg"
        all_files = glob.glob(pattern)
        files = []
        for file in all_files:
            speaker = int(file.split("/")[-2][1:])
            video_name = file.split("/")[-1][:-4]
            if speaker >= self.speakers[0] and speaker < self.speakers[1]:
                mouth_path = os.path.join(self.path, 'mouths', "s" + str(speaker), video_name)
                if not os.path.exists(mouth_path):
                    continue
                if len(os.listdir(mouth_path)) != 75:
                    continue

                align = os.path.join(self.path, 'aligns', "s" + str(speaker), video_name + '.align')
                sample = (file, align)
                files.append(sample)

        return files
