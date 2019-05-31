import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from head_pose import HeadPose


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    return frames


def save_frames(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    frames = extract_frames(video_path)
    for i, frame in enumerate(frames):
        path = os.path.join(output_path, str(i) + '.png')
        cv2.imwrite(path, frame)


def get_angles(video_path):
    head_pose = HeadPose()
    frames = extract_frames(video_path)
    angles = []
    for frame in frames:
        euler_angle = head_pose.predict(frame)
        angles.append(euler_angle)

    return angles


def plot_histogram(video_path):
    angles = get_angles(args.path)
    angles = np.array(angles)
    yaw_values = angles[:,1]
    plt.hist(yaw_values, bins=len(angles))
    plt.xlabel('Yaw')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()

    plot_histogram(args.path)
