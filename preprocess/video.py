import argparse
import os

import cv2


def extract_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_path, exist_ok=True)

    currentFrame = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        name = output_path + str(currentFrame) + '.jpg'
        cv2.imwrite(name, frame)
        currentFrame += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    extract_frames(args.path, output_path='data/frames/')
