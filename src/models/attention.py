import torch
from torch import nn


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
