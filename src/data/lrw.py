import os

import torch
from torch.utils.data import Dataset

from src.data.preprocess.lrw import load_video



class LRWDataset(Dataset):
    def __init__(self, directory, mode="train"):
        self.file_list = self.build_file_list(directory, mode)

    def build_file_list(self, directory, mode):
        labels = os.listdir(directory)
        videos = []

        for i, label in enumerate(labels):
            dirpath = directory + "/{}/{}".format(label, mode)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    filepath = dirpath + "/{}".format(file)
                    video = (i, filepath)
                    videos.append(video)

        return videos

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        label, filename = self.file_list[idx]
        frames = load_video(filename)
        sample = {'input': frames, 'label': torch.LongTensor([label])}
        return sample
