import argparse
import glob
import os

import cv2
import numpy as np
import psutil
import ray
from tqdm import tqdm

from face import FacePredictor, load_image
from video import load_mouth_images, load_video


@ray.remote
def preprocess_videos(videos, save_dir):
    face_predictor = FacePredictor()
    for video in videos:
        try:
            mouth_images = load_mouth_images(face_predictor, video, skip_frames=3)
        except Exception as e:
            print(e)
            continue
        video_name = video.split("/")[-1].rsplit(".mpg", 1)[0]
        speaker = video.split("/")[-2][1:]
        # save_path = save_dir + "/mouths/s" + speaker + "/" + video_name
        save_path = os.path.join(save_dir, "mouths", "s" + speaker, video_name)
        os.makedirs(save_path, exist_ok=True)
        for i, frame in enumerate(mouth_images):
            file = f'{save_path}/mouth_{i:03}.png'
            cv2.imwrite(file, frame)


def preprocess(directory):
    num_cpus = psutil.cpu_count()
    ray.init(num_cpus=num_cpus)

    save_dir = directory.rsplit("videos", 1)[0]
    pattern = directory + "/**/*.mpg"
    videos = glob.glob(pattern)
    splits = 200
    initial_split = np.array_split(videos, splits)
    with tqdm(total=len(videos)) as pbar:
        for split in range(splits):
            chunk = np.array_split(initial_split[split], num_cpus)
            ray.get([preprocess_videos.remote(chunk[i], save_dir) for i in range(num_cpus)])
            pbar.update(len(initial_split[split]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    preprocess(args.data)
