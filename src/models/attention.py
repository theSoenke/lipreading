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
        self.W = nn.Linear(attention_dim * num_experts, attention_dim * num_experts, bias=False)
        self.embedding = nn.Embedding(num_experts, attention_dim * num_experts)
        self.softmax = nn.Softmax(dim=1)

    def score(self, decoder_hidden, encoder_out):
        encoder_out = self.W(encoder_out)
        return encoder_out @ decoder_hidden

    def forward(self, decoder_hidden, encoder_out):
        embedding = self.embedding(decoder_hidden).transpose(1, 2)
        energies = self.score(embedding, encoder_out)
        mask = self.softmax(energies)
        context = encoder_out.transpose(2, 1) @ mask
        return context, mask
