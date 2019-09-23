import torch
from torch import nn


class FCAttention(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.input_dim = input_dim
        self.attn = nn.Linear(input_dim * num_experts, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.attn(x)
        x = self.relu(x)
        return x


class LuongAttention(nn.Module):
    def __init__(self, attention_dim, num_experts):
        super().__init__()
        self.W = nn.Linear(attention_dim, 1, bias=False)
        self.embedding = nn.Embedding(num_experts, 1)
        self.softmax = nn.Softmax(dim=1)

    def score(self, embedding, encoder_out):
        encoder_out = self.W(encoder_out)
        return encoder_out @ embedding

    def forward(self, degree, encoder_out):
        embedding = self.embedding(degree)
        energies = self.score(embedding, encoder_out)
        attn_score = self.softmax(energies)  # B x E x A
        return attn_score
