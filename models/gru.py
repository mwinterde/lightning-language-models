import torch
from torch import nn

from .base import BaseModule


class GRU(BaseModule):
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
        self.reset_gate = nn.Linear(
            self.embedding_dim + self.hidden_dim, self.hidden_dim
        )
        self.update_gate = nn.Linear(
            self.embedding_dim + self.hidden_dim, self.hidden_dim
        )
        self.cell_gate = nn.Linear(
            self.embedding_dim + self.hidden_dim, self.hidden_dim
        )
        self.linear_ay = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        device = x.device
        a = torch.zeros(B, self.hidden_dim, device=device)
        logits = torch.empty(B, T, self.vocab_size, device=device)
        for t in range(T):
            x_t = x[:, t]
            x_t = self.tok_embedding(x_t)
            r = torch.sigmoid(self.reset_gate(torch.cat([x_t, a], dim=1)))
            z = torch.sigmoid(self.update_gate(torch.cat([x_t, a], dim=1)))
            g = torch.tanh(self.cell_gate(torch.cat([x_t, r * a], dim=1)))
            a = (1 - z) * a + z * g
            logits[:, t, :] = self.linear_ay(a)
        return logits
