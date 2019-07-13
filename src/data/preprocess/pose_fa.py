import argparse
import math

import cv2
import face_alignment
import numpy as np


class HeadPose():
    def __init__(self, use_cuda=False):
        device = "cuda" if use_cuda else "cpu"
        self.predictor = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D,
            face_detector='sfd',
            flip_input=False,
            device=device,
            verbose=True,
        )

    def rotationToEuler(self, rotation):
        sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rotation[2, 1], rotation[2, 2])
            y = math.atan2(-rotation[2, 0], sy)
            z = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            x = math.atan2(-rotation[1, 2], rotation[1, 1])
            y = math.atan2(-rotation[2, 0], sy)
            z = 0

        return [x * 50, y * 50, z * 50]

    def predict(self, frame):
        points = self.predictor.get_landmarks_from_image(frame)
        if(points is None):
            return None

        points = points[0]
        rotation = np.empty((3, 3))
        rotation[0, :] = (points[16] - points[0])/np.linalg.norm(points[16] - points[0])
        rotation[1, :] = (points[8] - points[27])/np.linalg.norm(points[8] - points[27])
        rotation[2, :] = np.cross(rotation[0, :], rotation[1, :])

        angles = self.rotationToEuler(rotation)
        return {'pitch': angles[0], 'yaw': angles[1], 'roll': angles[2]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()

    head_pose = HeadPose(use_cuda=True)
    image = cv2.imread(args.path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angles = head_pose.predict(image)
    yaw = angles['yaw']

    print("Yaw: %f" % yaw)
