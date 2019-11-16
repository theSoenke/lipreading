import glob
import os

import cv2
import numpy as np
import psutil
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from tables import Float32Col, Int32Col, IsDescription, StringCol, open_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.ouluvs2 import OuluVS2Dataset
from src.preprocess.face_detection.facenet import FaceNet
from src.preprocess.head_pose.dlib_pose import HeadPose as DlibHeadPose
from src.preprocess.head_pose.face_alignment_pose import HeadPose as FaHeadPose
from src.preprocess.head_pose.hopenet import HeadPose as HopeNetHeadPose


def first_frame_tensor(video_path):
    video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
    frames = video.permute(0, 3, 1, 2)
    return transforms.functional.to_pil_image(frames[0])

def first_frame_cv2(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    cv2.destroyAllWindows()
    return frame


def build_file_list(path):
    videos = []
    pattern = path + "/**/*.mp4"
    files = glob.glob(pattern, recursive=True)
    for file in files:
        videos.append(file)
    return videos


def head_poses(path):
    predictor = DlibHeadPose()
    file_paths = build_file_list(path)
    size = 786
    degree = 20

    total_samples = {0: 0, 30: 0, 45: 0, 60: 0, 90: 0}
    correct_samples = {0: 0, 30: 0, 45: 0, 60: 0, 90: 0}
    for file in tqdm(file_paths):
        frame = first_frame_cv2(file)
        crop = (
            420,
            0,
            1500,
            1080,
        )

        split = file.split("/")[-1][:-4].split("_")
        speaker, view, utterance = [int(x[1:]) for x in split]
        view = [0, 30, 45, 60, 90][view-1]
        frame = Image.fromarray(frame)
        frame = frame.crop(crop).resize((256, 256))
        # frame = transforms.functional.to_tensor(frame)
        frame = np.asarray(frame)
        total_samples[view] += 1
        euler = predictor.predict(frame)
        yaw = abs(euler['yaw'])
        if yaw - degree <= view and yaw + degree >= view:
            correct_samples[view] += 1
        else:
            # print(f"Expected: {view:.2f}, Got: {yaw:.2f}, File: {file}")
            pass
    print(f"Correct samples: {correct_samples}")
    print(f"Samples per view: {total_samples}")

    correct = sum(correct_samples.values())
    acc = correct / (len(file_paths))
    print(f"Accuracy: {acc:.2f}")


class Video(IsDescription):
    yaw = Float32Col()
    view = Float32Col()
    speaker = Int32Col()
    utterance = Int32Col()
    file = StringCol(32)


def preprocess(path, output, workers=None):
    workers = psutil.cpu_count() if workers == None else workers
    if os.path.exists(output) == False:
        os.makedirs(output)

    output_path = "%s/ouluvs2.h5" % (output)
    if os.path.exists(output_path):
        os.remove(output_path)

    labels = None
    dataset = OuluVS2Dataset(path=path)
    preprocess_hdf5(
        dataset=dataset,
        output_path=output_path,
        table='train',
        workers=workers,
    )
    print("Saved preprocessed file: %s" % output_path)


def preprocess_hdf5(dataset, output_path, table, workers):
    file = open_file(output_path, mode="a")
    table = file.create_table("/", table, Video, expectedrows=len(dataset))
    row = table.row
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)

    with tqdm(total=len(dataset)) as progress:
        for batch in data_loader:
            for i in range(len(batch['yaw'])):
                for column in batch:
                    row[column] = batch[column][i]
                row.append()
                progress.update(1)
    table.flush()
    file.close()
