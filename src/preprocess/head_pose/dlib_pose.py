import argparse
import time

import cv2
import dlib
import numpy as np
from imutils import face_utils

# Pretrained model http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
mean_face = np.float32([[6.825897, 6.760612, 4.402142],
                        [1.330353, 7.122144, 6.903745],
                        [-1.330353, 7.122144, 6.903745],
                        [-6.825897, 6.760612, 4.402142],
                        [5.311432, 5.485328, 3.987654],
                        [1.789930, 5.393625, 4.413414],
                        [-1.789930, 5.393625, 4.413414],
                        [-5.311432, 5.485328, 3.987654],
                        [2.005628, 1.409845, 6.165652],
                        [-2.005628, 1.409845, 6.165652],
                        [2.774015, -2.080775, 5.048531],
                        [-2.774015, -2.080775, 5.048531],
                        [0.000000, -3.116408, 6.097667],
                        [0.000000, -7.415691, 4.070434]])


class HeadPose():
    def __init__(self, model_path='data/dlib/shape_predictor_68_face_landmarks.dat'):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(model_path)

    def predict(self, image):
        # face_position = self.face_detector(image, 1)
        # if len(face_position) == 0:
        #     return None
        # face_position = face_position[0]

        face_position = dlib.rectangle(0, 0, image.shape[0], image.shape[1])
        landmarks = self.landmark_predictor(image, face_position)
        landmarks = face_utils.shape_to_np(landmarks)
        image_pts = np.float32([landmarks[17], landmarks[21], landmarks[22], landmarks[26], landmarks[36],
                                landmarks[39], landmarks[42], landmarks[45], landmarks[31], landmarks[35],
                                landmarks[48], landmarks[54], landmarks[57], landmarks[8]])
        _, rotation_vec, translation_vec = cv2.solvePnP(mean_face, image_pts, cam_matrix, dist_coeffs)
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return {'pitch': euler_angle[0, 0], 'yaw': -euler_angle[1, 0], 'roll': euler_angle[2, 0]}

    def predict_from_path(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.predict(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()

    head_pose = HeadPose()
    start_time = time.time()
    for i in range(100):
        euler_angle = head_pose.predict_from_path(args.path)
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time / 100}s")
    print(euler_angle)

    print("Yaw: %f" % euler_angle['yaw'])
