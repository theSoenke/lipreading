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
from tqdm import tqdm

from src.data.preprocess.face import FacePredictor
from src.data.preprocess.facenet import FaceNet
from src.data.preprocess.video import load_video


class LRS2Dataset(Dataset):
    def __init__(self, path, in_channels=1, mode="train", augmentations=False, estimate_pose=False, max_timesteps=155, pretrain_words=0):
        self.max_timesteps = max_timesteps
        self.pretrain = mode == "pretrain"
        self.in_channels = in_channels
        self.estimate_pose = estimate_pose
        self.max_timesteps = max_timesteps
        self.pretrain_words = pretrain_words

        self.augmentation = augmentations if mode == 'train' or mode == "pretrain" else False
        self.file_paths, self.file_names = self.build_file_list(path, mode)

        numbers = "".join([str(i) for i in range(10)])
        special_characters = " '"
        self.characters = special_characters + ascii_lowercase + numbers
        int2char = dict(enumerate(self.characters))
        self.char2int = {char: index for index, char in int2char.items()}

    def build_file_list(self, directory, mode):
        file_list = []
        paths = []
        if self.pretrain:
            file = open(f"{directory}/pretrain.txt", "r")
            content = file.read()
            for file in content.splitlines():
                file_list.append(file)
                paths.append(f"{directory}/mvlrs_v1/pretrain/{file}")
        else:
            file = open(f"{directory}/{mode}.txt", "r")
            content = file.read()
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

    def get_pretrain_words(self, fps, content, file_name):
        lines = content.splitlines()[4:]
        words = []
        for line in lines:
            word, start, stop, _ = line.split(" ")
            start, stop = float(start), float(stop)
            words.append([word, start, stop])

        num_words = min(random.randint(1, self.pretrain_words), len(words))
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
        start_frame = int(sample_start * fps)
        stop_frame = math.ceil(sample_end * fps)

        if stop_frame - start_frame > self.max_timesteps:
            print(f"Cutting frames off. Requires {stop_frame - start_frame} frames: {file_name}")
            stop_frame = start_frame + self.max_timesteps

        return content.strip(), start_frame, stop_frame

    def __getitem__(self, idx):
        file = self.file_names[idx]
        video, _, info = torchvision.io.read_video(self.file_paths[idx] + ".mp4")  # T, H, W, C
        file_path = self.file_paths[idx]
        content = open(file_path + ".txt", "r").read()

        if self.pretrain:
            fps = info['video_fps']
            content, start_frame, stop_frame = self.get_pretrain_words(fps, content, file_path)
            video = video[start_frame:stop_frame]
        else:
            content = content.splitlines()[0][7:]

        assert len(video) <= self.max_timesteps, f"Video too large with {len(video)} frames: {file_path}"
        content = content.lower()
        frames = self.build_tensor(video)
        encoded = [self.char2int[char] for char in content]
        input_lengths = video.size(0)
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
                frames, _, input_lengths, ids = batch
                for i, video in enumerate(frames):
                    video = video.transpose(1, 0)[:input_lengths[i]]  # T C H W
                    yaws = head_pose.predict(video)['yaw']
                    yaws = ";".join([f"{yaw:.2f}" for yaw in yaws.cpu().numpy()])
                    line = f"{dataset.file_names[ids[i]]};{yaws}"
                    lines.append(line)
                    progress.update(1)
        file = open(f"{output_path}/{mode}.txt", "w")
        file.write('\n'.join(lines))
        file.close()


class LRS2DatasetMouth(Dataset):
    def __init__(self, path, mode="train"):
        self.file_paths, self.file_names = self.build_file_list(path, mode)
        # self.facenet = FaceNet()
        self.predictor = FacePredictor()

    def build_file_list(self, directory, mode):
        file_list = []
        paths = []

        if mode == "pretrain":
            file = open(f"{directory}/pretrain.txt", "r")
            content = file.read()
            for file in content.splitlines():
                file_list.append(f"pretrain/{file}")
                paths.append(f"{directory}/mvlrs_v1/pretrain/{file}")
        else:
            file = open(f"{directory}/{mode}.txt", "r")
            content = file.read()
            for file in content.splitlines():
                file = file.split(" ")[0]
                file_list.append(f"main/{file}")
                paths.append(f"{directory}/mvlrs_v1/main/{file}")

        return paths, file_list

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        video_path = self.file_paths[idx] + ".mp4"
        video, _, _ = torchvision.io.read_video(video_path)
        frames = video.permute(0, 3, 1, 2)  # T C H W

        skip = False
        boxes = []
        for i, frame in enumerate(frames):
            bb = self.facenet.detect(frame)
            if bb is None:
                print(f"No face in frame {i} detected: {file_name}")
                skip = True
                break
            if len(bb) > 1:
                box = bb[0]  # first is largest face
            else:
                box = bb[0]

            box = [str(f"{pos:.2f}") for pos in box]
            boxes.append(";".join(box))

        return {'bb': boxes, 'file': file_name, 'skip': skip}


def mouth_bounding_boxes(path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for mode in ['val', 'test', 'train', 'pretrain']:
        dataset = LRS2DatasetMouth(path=path, mode=mode)
        lines = []
        with tqdm(total=len(dataset)) as progress:
            for sample in dataset:
                skip = sample['skip']
                if not skip:
                    boxes = sample['bb']
                    file = sample['file']
                    boxes = "|".join(boxes)
                    line = f"{file};{boxes}"
                    lines.append(line)
                progress.update(1)

        file = open(f"{output_path}/{mode}_face_boxes.txt", "w")
        file.write('\n'.join(lines))
        file.close()
