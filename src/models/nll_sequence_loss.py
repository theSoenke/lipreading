from torch import nn


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
