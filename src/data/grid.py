import glob
import os

from torch.utils.data import Dataset

from src.data.preprocess.video import load_video


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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        frames = load_video(file)
        return {"frames": frames, "text": "test"}

    def build_file_list(self):
        pattern = self.path + "/**/*.mpg"
        all_files = glob.glob(pattern)
        files = []
        for file in all_files:
            speaker = int(file.split("/")[-2][1:])
            if speaker >= self.speakers[0] and speaker < self.speakers[1]:
                files.append(file)

        return files
