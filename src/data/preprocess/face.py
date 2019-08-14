import argparse
import os

import cv2
import dlib


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


class FacePredictor():
    def __init__(self, model_path='data/dlib/shape_predictor_68_face_landmarks.dat'):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def face_rect(self, image, path=None):
        faces = self.detector(image, 1)
        assert len(faces) == 1, "Expected 1 face, got %d for %s" % (len(faces), path)
        return faces[0]

    def mouth_image(self, image, path=None):
        face_rect = self.face_rect(image, path)
        return self.mouth_image_rect(image, face_rect)

    def mouth_image_rect(self, image, face_rect):
        pad = 10
        width = 40
        height = 20

        shape = self.predictor(image, face_rect)
        xmouthpoints = [shape.part(x).x for x in range(48, 67)]
        ymouthpoints = [shape.part(x).y for x in range(48, 67)]
        maxx = max(xmouthpoints)
        minx = min(xmouthpoints)
        maxy = max(ymouthpoints)
        miny = min(ymouthpoints)

        crop_image = image[miny-pad:miny+height+pad, minx-pad:minx+width+pad]
        assert crop_image.size == (width + 2 * pad) * (height + 2 * pad)
        return crop_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='data/pretrained/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--image')
    args = parser.parse_args()
    image_path = args.image

    face_predictor = FacePredictor(args.model)
    image = load_image(image_path)
    mouth_image = face_predictor.mouth_image(image)
    os.makedirs("data/mouths/", exist_ok=True)
    cv2.imwrite("data/mouths/mouth.png", mouth_image)
