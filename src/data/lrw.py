import os

import imageio
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.preprocess.head_pose import HeadPose


class LRWDataset(Dataset):
    def __init__(self, directory, num_words=500, mode="train"):
        self.num_words = num_words
        self.file_list, self.labels = self.build_file_list(directory, mode)
        self.head_pose = HeadPose()

    def build_file_list(self, directory, mode):
        labels = os.listdir(directory)[:self.num_words]
        print(labels)
        videos = []

        for i, label in enumerate(labels):
            dirpath = directory + "/{}/{}".format(label, mode)
            files = os.listdir(dirpath)
            for file in files:
                if file.endswith("mp4"):
                    filepath = dirpath + "/{}".format(file)
                    video = (i, filepath)
                    videos.append(video)

        return videos, labels


    def load_video(self, file):
        vid = imageio.get_reader(file,  'ffmpeg')
        frames = []
        for i in range(0, 29):
            image = vid.get_data(i)
            if i == 14:
                try:
                    angles = self.head_pose.predict(image)
                    yaw = -angles[1, 0]
                except:
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
        label, filename = self.file_list[idx]
        frames, yaw = self.load_video(filename)
        if yaw == None:
            yaw = 1000
            # print("No face found: %s" % filename)
        sample = {'input': frames, 'label': torch.LongTensor([label]), 'yaw': yaw}
        return sample
