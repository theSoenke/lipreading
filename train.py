import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.grid import GridDataset

epochs = 10
learning_rate = 1e-3
batch_size = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data')
args = parser.parse_args()
data_path = args.data

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = DataLoader(GridDataset(path=data_path), shuffle=True, batch_size=batch_size, num_workers=4)

