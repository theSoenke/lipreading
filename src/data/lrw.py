import os
import random

import psutil
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm

from src.data.transforms import StatefulRandomHorizontalFlip


def build_word_list(directory, num_words, seed):
    random.seed(seed)
    words = os.listdir(directory)
    words.sort()
    random.shuffle(words)
    words = words[:num_words]
    return words


class LRWDataset(Dataset):
    def __init__(self, path, num_words=500, mode="train", augmentation=False, estimate_pose=False, seed=42, query=None):
        self.seed = seed
        self.num_words = num_words
        self.query = query  # FIXME
        self.augmentation = augmentation if mode == 'train' else False
        self.poses = self.head_poses(mode, query)
        video_paths, self.files, self.labels, self.words = self.build_file_list(path, mode)
        self.video_clips = VideoClips(
            video_paths,
            clip_length_in_frames=29,
            # num_workers=4,
        )
        self.estimate_pose = estimate_pose

    def head_poses(self, mode, query):
        poses = {}
        yaw_file = open(f"data/preprocessed/{mode}.txt", "r")
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
                if file in self.poses:
                    path = dirpath + "/{}".format(file)
                    file_list.append(file)
                    paths.append(path)
                    labels.append(i)

        return paths, file_list, labels, words

    def build_tensor(self, frames):
        temporalVolume = torch.FloatTensor(1, 29, 112, 112)
        if(self.augmentation):
            augmentations = transforms.Compose([
                StatefulRandomHorizontalFlip(0.5),
            ])
        else:
            augmentations = transforms.Compose([])

        for i in range(0, 29):
            frame = frames[i].permute(2, 0, 1)  # (Tensor[C, H, W])
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((112, 112)),
                augmentations,
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.4161, ], [0.1688, ]),
            ])(frame)
            temporalVolume[0][i] = result

        return temporalVolume

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        label = self.labels[idx]
        file = self.files[idx]
        video, _, _, _ = self.video_clips.get_clip(idx)  # (Tensor[T, H, W, C])
        if self.estimate_pose:
            angle_frame = video[14].permute(2, 0, 1)
        else:
            angle_frame = 0
        frames = self.build_tensor(video)
        sample = {
            'frames': frames,
            'label': torch.LongTensor([label]),
            'word': self.words[label],
            'file': self.files[idx],
            'yaw': self.poses[file],
            'angle_frame': angle_frame,
        }
        return sample


def extract_angles(path, output_path, num_workers, seed):
    from src.data.preprocess.pose_hopenet import HeadPose
    head_pose = HeadPose()

    words = None
    for mode in ['train', 'val']:
        dataset = LRWDataset(directory=path, num_words=500, mode=mode, estimate_pose=True, seed=seed)
        if words != None:
            assert words == dataset.words
        words = dataset.words
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
        lines = ""
        with tqdm(total=len(dataset)) as progress:
            for batch in data_loader:
                frames = batch['angle_frame']
                files = batch['file']
                yaws = head_pose.predict(frames)['yaw']
                for i in range(len(batch['frames'])):
                    line = f"{files[i]},{yaws[i].item():.2f}\n"
                    lines += line
                    progress.update(1)
        file = open(f"{output_path}/{mode}.txt", "w")
        file.write(lines)
        file.close()


class Video(IsDescription):
    label = Int32Col()
    frames = Float32Col(shape=(29, 112, 112))
    yaw = Float32Col()
    file = StringCol(32)
    word = StringCol(32)


def preprocess(path, output, num_words, augmentation=False, workers=None):
    workers = psutil.cpu_count() if workers == None else workers
    if os.path.exists(output) == False:
        os.makedirs(output)

    if augmentation:
        output_path = "%s/lrw_aug_%d.h5" % (output, num_words)
    else:
        output_path = "%s/lrw_%d.h5" % (output, num_words)
    if os.path.exists(output_path):
        os.remove(output_path)

    words = None
    for mode in ['train', 'val', 'test']:
        print("Generating %s data" % mode)
        dataset = LRWDataset(directory=path, num_words=num_words, mode=mode, augmentation=augmentation, estimate_pose=True)
        if words != None:
            assert words == dataset.words
        words = dataset.words
        preprocess_hdf5(
            dataset=dataset,
            output_path=output_path,
            table=mode,
            workers=workers,
        )
    print("Saved preprocessed file: %s" % output_path)


def preprocess_hdf5(dataset, output_path, table, workers=0):
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video, expectedrows=len(dataset))
    row = table.row
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['yaw'])):
                for column in batch:
                    value = batch[column][i]
                    if isinstance(value, str):
                        row[column] = batch[column][i]
                    else:
                        row[column] = batch[column][i].numpy()
                row.append()
                progress.update(1)
    table.flush()
    file.close()
