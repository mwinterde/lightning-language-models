import torch
from torch import nn

from .base import BaseModule


class Transformer(BaseModule):
    num_heads = 4
    embedding_dim = 32
    dropout = 0.1

    def __init__(self, vocab_size, context, learning_rate, tokenizer):
        super().__init__(
            vocab_size=vocab_size,
            context=context,
            learning_rate=learning_rate,
            tokenizer=tokenizer,
        )
        self.tok_embedding = torch.nn.Embedding(vocab_size, self.embedding_dim)
        self.pos_embedding = torch.nn.Embedding(context, self.embedding_dim)
        self.encoders = torch.nn.Sequential(
            *[
                EncoderBlock(
                    context=context,
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.fc = torch.nn.Linear(self.embedding_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        device = x.device
        positions = torch.arange(T, device=device)
        x = self.tok_embedding(x) + self.pos_embedding(positions)
        x = self.encoders(x)
        logits = self.fc(x)
        return logits


class EncoderBlock(torch.nn.Module):
    def __init__(self, context, embedding_dim, num_heads, dropout):
        super().__init__()
        self.attention = SelfAttention(
            context, embedding_dim, num_heads, dropout
        )
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * num_heads),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * num_heads, embedding_dim),
        )
        self.res_dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm1(x + self.attention(x))
        y = self.norm2(y + self.ff(y))
        y = self.res_dropout(y)
        return y


class SelfAttention(torch.nn.Module):
    def __init__(self, context, embedding_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.qkv = torch.nn.Linear(embedding_dim, embedding_dim * 3)
        self.fc = torch.nn.Linear(embedding_dim, embedding_dim)
        self.att_dropout = torch.nn.Dropout(dropout)
        self.res_dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context, context)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / (self.head_size**0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.att_dropout(att)
        y = att @ v
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, C)
        y = self.fc(y)
        y = self.res_dropout(y)
        return y
