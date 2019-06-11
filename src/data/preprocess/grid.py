import argparse
import glob

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
        mouth_images = load_mouth_images(face_predictor, video, skip_frames=3)
        video_name = video.split("/")[-1].rsplit(".mpg", 1)[0]
        speaker = video.split("/")[-2][1:]
        save_path = save_dir + "/mouths/s" + speaker + "/" + video_name
        for i, frame in enumerate(mouth_images):
            cv2.imwrite(save_path + "/" + str(i) + ".png", frame)


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
