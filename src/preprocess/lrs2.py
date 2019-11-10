import os

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.lrs2 import LRS2Dataset
from src.data.transforms import Crop
from src.preprocess.face_detection.facenet import FaceNet
from src.preprocess.head_pose.hopenet import HeadPose


def extract_angles(path, output_path, num_workers):
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
    def __init__(self, path, mode="train", skip_frames=1):
        self.skip_frames = skip_frames
        self.file_paths, self.file_names = self.build_file_list(path, mode)
        self.facenet = FaceNet()
        # torchvision.set_video_backend('video_reader')

    def build_file_list(self, directory, mode):
        file_list = []
        paths = []

        if mode == "pretrain":
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

    def __len__(self):
        return len(self.file_names)

    def extract_bb(self, landmarks):
        left = int(landmarks[3])
        upper = int(landmarks[8])
        right = int(landmarks[4])
        lower = int(landmarks[9])
        return left, upper, right, lower

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        video_path = self.file_paths[idx] + ".mp4"
        video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        frames = video.permute(0, 3, 1, 2)  # T C H W
        num_frames = len(video)

        every_nth_frame = []
        for i, frame in enumerate(frames):
            if i % self.skip_frames == 0:
                every_nth_frame.append(frame)

        boxes = []
        try:
            _, batch_landmarks = self.facenet.detect(every_nth_frame)
        except Exception as e:
            print(f"Could not process: {file_name}", e)
            return {'bb': [], 'file': file_name, 'skip': True}

        if len(every_nth_frame) != len(batch_landmarks):
            print("Mismatch of detected landmarks")
            return {'bb': [], 'file': file_name, 'skip': True}

        for landmarks in batch_landmarks:
            if len(landmarks) == 0 or landmarks.shape[1] == 0 or landmarks.shape[0] == 0:
                print(f"No face found: {video_path}")
                return {'bb': [], 'file': file_name, 'skip': True}

            if landmarks.shape[1] >= 2:
                # choose largest face
                selected = 0
                max_size = 0
                for i in range(landmarks.shape[1]):
                    left, upper, right, lower = self.extract_bb(landmarks[:, i])
                    size = (right - left) * (lower - upper)
                    if size > max_size:
                        max_size = size
                        selected = i
                landmarks = landmarks[:, selected]

            width = 96
            height = 64

            left, upper, right, lower = self.extract_bb(landmarks)
            vertical_center = (left + right) / 2
            horizontal_center = (upper + lower) / 2

            box = [
                horizontal_center - (width // 2),
                vertical_center - (height // 2),
                horizontal_center + (width // 2),
                vertical_center + (height // 2),
            ]

            box = [str(f"{pos}") for pos in box]
            boxes.append(";".join(box))

        all_boxes = []
        for box in boxes:
            for i in range(self.skip_frames):
                if len(all_boxes) == num_frames:
                    break
                else:
                    all_boxes.append(box)

        assert len(all_boxes) == num_frames
        return {'bb': all_boxes, 'file': file_name, 'skip': False}


def mouth_bounding_boxes(path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for mode in ['val', 'test', 'train', 'pretrain']:
        dataset = LRS2DatasetMouth(path=path, mode=mode, skip_frames=5)
        lines = []
        with tqdm(total=len(dataset)) as progress:
            progress.set_description(mode)
            for sample in dataset:
                skip = sample['skip']
                box = sample['bb']
                file = sample['file']

                if not skip:
                    box = "|".join(box)
                    line = f"{file}:{box}"
                    lines.append(line)
                progress.update(1)

        file = open(f"{output_path}/{mode}_crop.txt", "w")
        file.write('\n'.join(lines))
        file.close()


def prepare_language_model(path, output_path):
    os.makedirs(output_path, exist_ok=True)

    file = open(f"{path}/train.txt", "r")
    content = file.read()
    sentence_lines = []
    # char_lines = []
    for file in content.splitlines():
        label_file = open(f"{path}/mvlrs_v1/main/{file}.txt")
        label = label_file.read()
        sentence = label.splitlines()[0][7:].lower()
        sentence_lines.append(sentence)
        # sentence = sentence.replace(' ', '@') # prepare for training char lm
        # char_lines.append(' '.join([char for char in sentence]))
    file = open(f"{output_path}/sentences.txt", "w")
    file.write('\n'.join(sentence_lines))
    file.close()

    # file = open(f"{output_path}/characters.txt", "w")
    # file.write('\n'.join(char_lines))
    # file.close()
