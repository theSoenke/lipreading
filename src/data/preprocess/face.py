import argparse
import os

import cv2
import dlib


def detect_face(detector, image_path):
    print("Processing file: {}".format(image_path))
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(img, 1)
    assert len(faces) == 1, "Expected 1 face, got %d" % len(faces)

    face = faces[0]
    print("Left: {} Top: {} Right: {} Bottom: {}".format(face.left(), face.top(), face.right(), face.bottom()))
    return img, face


def detect_mouth(predictor, img, face_rect):
    pad = 10
    width = 40
    height = 20

    shape = predictor(img, face_rect)
    xmouthpoints = [shape.part(x).x for x in range(48, 67)]
    ymouthpoints = [shape.part(x).y for x in range(48, 67)]
    maxx = max(xmouthpoints)
    minx = min(xmouthpoints)
    maxy = max(ymouthpoints)
    miny = min(ymouthpoints)

    crop_image = img[miny-pad:miny+height+pad, minx-pad:minx+width+pad]
    return crop_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='data/pretrained/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--image')
    args = parser.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model)

    face_img, face = detect_face(detector, args.image)
    mouth_img = detect_mouth(predictor, face_img, face)
    os.makedirs("data/mouths/", exist_ok=True)
    cv2.imwrite("data/mouths/mouth.png", mouth_img)
