import argparse
import math

import cv2
import face_alignment
import numpy as np

# https://gist.github.com/zalo/71fbd5dbfe23cb46406d211b84be9f7e


class HeadPose():
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device="cuda")

    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z]) * 100

    def predict(self, frame):
        imagePoints = self.fa.get_landmarks_from_image(frame)
        if(imagePoints is not None):
            imagePoints = imagePoints[0]

            # Compute the Mean-Centered-Scaled Points
            mean = np.mean(imagePoints, axis=0)  # <- This is the unscaled mean
            scaled = (imagePoints / np.linalg.norm(imagePoints[42] - imagePoints[39])) * 0.06  # Set the inner eye distance to 60cm (just because)
            centered = scaled - np.mean(scaled, axis=0)  # <- This is the scaled mean

            # Construct a "rotation" matrix (strong simplification, might have shearing)
            rotationMatrix = np.empty((3, 3))
            rotationMatrix[0, :] = (centered[16] - centered[0])/np.linalg.norm(centered[16] - centered[0])
            rotationMatrix[1, :] = (centered[8] - centered[27])/np.linalg.norm(centered[8] - centered[27])
            rotationMatrix[2, :] = np.cross(rotationMatrix[0, :], rotationMatrix[1, :])
            invRot = np.linalg.inv(rotationMatrix)

            print(self.rotationMatrixToEulerAngles(rotationMatrix))

            # Object-space points, these are what you'd run OpenCV's solvePnP() with
            objectPoints = centered.dot(invRot)

            # Draw the computed data
            for i, (imagePoint, objectPoint) in enumerate(zip(imagePoints, objectPoints)):
                # Draw the Point Predictions
                cv2.circle(frame, (imagePoint[0], imagePoint[1]), 3, (0, 255, 0))

                # Draw the X Axis
                cv2.line(frame, tuple(mean[:2].astype(int)), tuple((mean+(rotationMatrix[0, :] * 100.0))[:2].astype(int)), (0, 0, 255), 3)
                # Draw the Y Axis
                cv2.line(frame, tuple(mean[:2].astype(int)), tuple((mean-(rotationMatrix[1, :] * 100.0))[:2].astype(int)), (0, 255, 0), 3)
                # Draw the Z Axis
                cv2.line(frame, tuple(mean[:2].astype(int)), tuple((mean+(rotationMatrix[2, :] * 100.0))[:2].astype(int)), (255, 0, 0), 3)

        cv2.imshow('View', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()

    head_pose = HeadPose()
    image = cv2.imread(args.path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    head_pose.predict(image)
    # euler_angle =
    # print(euler_angle)
    # yaw = -euler_angle[1, 0]
    # pitch = euler_angle[0, 0]
    # roll = euler_angle[2, 0]

    # print("Yaw: %f" % yaw)
