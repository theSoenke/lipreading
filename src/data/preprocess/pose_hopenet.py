import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.hopenet.hopenet import Hopenet


class HeadPose():
    def __init__(self, checkpoint='data/hopenet/hopenet_robust_alpha1.pkl', transform=None):
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        num_bins = 66
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.idx_tensor = torch.FloatTensor([idx for idx in range(num_bins)]).to(self.device)
        self.model = Hopenet()
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, list):
            image = [self.transform(img) for img in image]
        elif isinstance(image, str):
            image = Image.open(image)
            image = self.transform(image).unsqueeze(dim=0)
        else:
            image = self.transform(image).unsqueeze(dim=0)

        image = image.to(self.device)
        yaw, pitch, roll = self.model(image)
        yaw = F.softmax(yaw, dim=1)
        pitch = F.softmax(pitch, dim=1)
        roll = F.softmax(roll, dim=1)

        yaw = torch.sum(yaw * self.idx_tensor, dim=1) * 3 - 99
        pitch = torch.sum(pitch * self.idx_tensor, dim=1) * 3 - 99
        roll = torch.sum(roll * self.idx_tensor, dim=1) * 3 - 99
        return {'yaw': yaw.item(), 'pitch': pitch.item(), 'roll': roll.item()}
