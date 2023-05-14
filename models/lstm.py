import torch
from torch import nn

from .base import BaseModule


class LSTM(BaseModule):
    embedding_dim = 32
    hidden_dim = 128

    def __init__(self, vocab_size, context, learning_rate, tokenizer):
        super().__init__(
            vocab_size=vocab_size,
            context=context,
            learning_rate=learning_rate,
            tokenizer=tokenizer,
        )
        self.tok_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.forget_gate = nn.Linear(
            self.embedding_dim + self.hidden_dim, self.hidden_dim
        )
        self.input_gate = nn.Linear(
            self.embedding_dim + self.hidden_dim, self.hidden_dim
        )
        self.output_gate = nn.Linear(
            self.embedding_dim + self.hidden_dim, self.hidden_dim
        )
        self.cell_gate = nn.Linear(
            self.embedding_dim + self.hidden_dim, self.hidden_dim
        )
        self.linear_ay = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        a = torch.zeros(B, self.hidden_dim)
        c = torch.zeros(B, self.hidden_dim)
        logits = torch.empty(B, T, self.vocab_size)
        for t in range(T):
            x_t = x[:, t]
            x_t = self.tok_embedding(x_t)
            x_t = torch.cat([x_t, a], dim=1)
            f = torch.sigmoid(self.forget_gate(x_t))
            i = torch.sigmoid(self.input_gate(x_t))
            o = torch.sigmoid(self.output_gate(x_t))
            g = torch.tanh(self.cell_gate(x_t))
            c = f * c + i * g
            a = o * torch.tanh(c)
            logits[:, t, :] = self.linear_ay(a)
        return logits
