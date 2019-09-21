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

    def score(self, embedding, encoder_out):
        embedding = self.W(embedding)
        return embedding @ encoder_out.transpose(1, 2)

    def forward(self, degree, encoder_out):
        embedding = self.embedding(degree)
        energies = self.score(embedding, encoder_out)
        mask = self.softmax(energies)
        context = embedding.transpose(2, 1) @ mask
        return context.transpose(1, 2), mask


class LuongAttention2(nn.Module):
    def __init__(self, attention_dim, num_experts):
        super().__init__()
        self.W = nn.Linear(attention_dim * num_experts, attention_dim * num_experts, bias=False)
        self.embedding = nn.Embedding(num_experts, attention_dim * num_experts)
        self.softmax = nn.Softmax(dim=1)

    def score(self, embedding, encoder_out):
        import pdb
        pdb.set_trace()
        embedding = self.W(embedding)
        return embedding @ encoder_out.transpose(1, 2)

    def forward(self, degree, encoder_out):
        embedding = self.embedding(degree)
        energies = self.score(embedding, encoder_out)
        attn_score = self.softmax(energies)
        return attn_score


class LuongAttention3(nn.Module):
    def __init__(self, attention_dim, num_experts):
        super().__init__()
        self.W = nn.Linear(attention_dim * num_experts, attention_dim * num_experts, bias=False)
        self.embedding = nn.Embedding(num_experts, attention_dim * num_experts)
        self.softmax = nn.Softmax(dim=1)

    def score(self, embedding, encoder_out):
        embedding = self.W(embedding)
        return embedding @ encoder_out.transpose(1, 2)

    def forward(self, degree, encoder_out):
        import pdb
        pdb.set_trace()
        embedding = self.embedding(degree)
        energies = self.score(embedding, encoder_out)
        attn_score = self.softmax(energies)
        return attn_score
