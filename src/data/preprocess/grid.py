import argparse
import glob

import cv2
from tqdm import tqdm

from face import FacePredictor, load_image
from video import load_mouth_images, load_video


def preprocess(directory):
    save_dir = directory.rsplit("videos", 1)[0]
    face_predictor = FacePredictor()
    pattern = directory + "/**/*.mpg"
    videos = glob.glob(pattern)
    for video in tqdm(videos):
        mouth_images = load_mouth_images(face_predictor, video, skip_frames=3)
        video_name = video.split("/")[-1].rsplit(".mpg", 1)[0]
        speaker = video.split("/")[-2][1:]
        save_path = save_dir + "/mouths/s" + speaker + "/" + video_name
        for i, frame in enumerate(mouth_images):
            cv2.imwrite(save_path + "/" + str(i) + ".png", frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    preprocess(args.data)
