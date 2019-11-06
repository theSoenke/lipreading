import argparse

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms


class FaceNet():
    def __init__(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(select_largest=True, device=device)

    def detect(self, image):
        if isinstance(image, torch.Tensor) and len(image.shape) == 4:
            data = []
            for frame in image:
                frame = transforms.functional.to_pil_image(frame)
                data.append(frame)
        else:
            data = transforms.functional.to_pil_image(image)

        boxes, _ = self.mtcnn.detect(data)
        return boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()
    image_path = args.image

    facenet = FaceNet()
    image = Image.open(image_path)
    bb = facenet.detect(image)
    print(bb)
