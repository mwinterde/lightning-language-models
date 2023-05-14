from torch import nn

from .base import BaseModule


class Bigram(BaseModule):
    def __init__(self, vocab_size, context, learning_rate, tokenizer):
        super().__init__(
            vocab_size=vocab_size,
            context=context,
            learning_rate=learning_rate,
            tokenizer=tokenizer,
        )
        self.tok_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.tok_embedding(x)
