import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, attention_dim, num_experts):
        super().__init__()
        self.fc1 = nn.Linear(1, attention_dim)
        self.fc2 = nn.Linear(attention_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, degree):
        x = self.fc1(degree)
        x = self.fc2(x)
        attn_score = self.softmax(x)  # B x E x A
        return attn_score
