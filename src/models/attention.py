import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class LuongAttention():
    def __init__(self, input_dim, num_experts):
        self.input_dim = input_dim
        self.attn = nn.Linear(input_dim, input_dim)
        self.embedding = nn.Embedding(num_experts, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, experts_hidden, degree):
        embedded = self.embedding(degree).view(1, 1, -1)
        x = torch.cat(experts_hidden, dim=1)
        attn_weights = self.softmax(self.attn(torch.cat((embedded[0], x[0]), dim=1)), dim=1)

        attn = self.attn(x)
        attn_weights = self.softmax(attn, dim=1)
        _, indices = torch.max(attn_weights, dim=1)
        # max weights of experts
        x = None

        return x
