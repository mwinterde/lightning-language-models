import torch
from torch import nn

from .base import BaseModule


class RNN(BaseModule):
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
        self.linear_xa = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear_aa = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_ay = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        device = x.device
        a = torch.zeros(B, self.hidden_dim, device=device)
        logits = torch.empty(B, T, self.vocab_size, device=device)
        for t in range(T):
            x_t = x[:, t]
            x_t = self.tok_embedding(x_t)
            x_t = self.linear_xa(x_t)
            a = self.linear_aa(a)
            a = torch.tanh(x_t + a)
            logits[:, t, :] = self.linear_ay(a)
        return logits
