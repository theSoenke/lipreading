from torch import nn
from torch.functional import F

from src.models.resnet import ResNetModel


class NLLSequenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.NLLLoss()

    def forward(self, pred, target):
        loss = 0.0
        transposed = pred.transpose(0, 1).contiguous()
        num_frames = pred.shape[1]
        for i in range(num_frames):
            loss += self.criterion(transposed[i], target)

        return loss


class Model(nn.Module):
    def __init__(self, num_classes, resnet_layers=18, resnet_pretrained=False):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = ResNetModel(layers=resnet_layers, pretrained=resnet_pretrained)
        self.lstm = self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256 * 2, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)
        # self.loss = NLLSequenceLoss()
        self.loss = nn.CrossEntropyLoss()
        self.backend = ConvBackend(num_classes)

    def forward(self, x):
        x = self.frontend(x)
        x = self.resnet(x)
        x = self.backend(x)
        # x, _ = self.lstm(x)
        # x = self.fc(x)
        # x = self.softmax(x)
        return x


class ConvBackend(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        bn_size = 256
        self.conv1 = nn.Conv1d(bn_size, 2 * bn_size, 2, 2)
        self.norm1 = nn.BatchNorm1d(bn_size * 2)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(2 * bn_size, 4 * bn_size, 2, 2)
        self.norm2 = nn.BatchNorm1d(bn_size * 4)

        self.linear = nn.Linear(4*bn_size, bn_size)
        self.norm3 = nn.BatchNorm1d(bn_size)
        self.linear2 = nn.Linear(bn_size, num_classes)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        transposed = input.transpose(1, 2).contiguous()

        output = self.conv1(transposed)
        output = self.norm1(output)
        output = F.relu(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = F.relu(output)
        output = output.mean(2)
        output = self.linear(output)
        output = self.norm3(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output
