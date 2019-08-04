from torch import nn

from src.models.resnet import ResNetModel
from src.models.time_batch_wrapper import TimeBatchWrapper


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
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256 * 2, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)
        self.loss = NLLSequenceLoss()
        # self.pred = nn.Sequential(
        #     TimeBatchWrapper(mod=nn.Linear(256 * 2, 26 + 1))
        #     # T B V'
        # )

    def forward(self, x):
        x = self.frontend(x)
        x = self.resnet(x)
        x, _ = self.lstm(x)
        # x = self.pred(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
