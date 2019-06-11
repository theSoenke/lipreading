import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from head_pose import HeadPose


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    return frames


def save_frames(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    frames = load_video(video_path)
    for i, frame in enumerate(frames):
        path = os.path.join(output_path, str(i) + '.png')
        cv2.imwrite(path, frame)


def extract_angles(video_path):
    head_pose = HeadPose()
    frames = load_video(video_path)
    angles = []
    for frame in frames:
        euler_angle = head_pose.predict(frame)
        angles.append(euler_angle)

    return angles


def load_mouth_images(predictor, video_path, skip_frames=3):
    frames = load_video(video_path)
    face_rect = None
    mouth_frames = []
    for i, frame in enumerate(frames):
        if i % skip_frames == 0:
            face_rect = predictor.face_rect(frame, video_path)
        mouth_image = predictor.mouth_image_rect(frame, face_rect)
        mouth_frames.append(mouth_image)

    return mouth_frames


def plot_histogram(video_path):
    angles = extract_angles(args.path)
    angles = np.array(angles)
    yaw_values = angles[:, 1]
    plt.hist(yaw_values, bins=len(angles))
    plt.xlabel('Yaw')
    plt.show()


def process_directory(directory, file_type='mp4'):
    pattern = directory + "/*.%s" % file_type
    videos = glob.glob(pattern)
    print("Processing  %d videos" % len(videos))

    for i, video in enumerate(videos):
        angles = extract_angles(video)
        print("[%d/%d]: %s" % (i, len(videos), video))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()

    process_directory(args.path, 'mpg')

    # plot_histogram(args.path)
