import lightning.pytorch as pl
from torch import nn, optim


class BaseModule(pl.LightningModule):
    def __init__(self, vocab_size, context, learning_rate, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.context = context
        self.learning_rate = learning_rate
        self.tokenizer = tokenizer

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.calculate_loss(x, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.calculate_loss(x, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def calculate_loss(self, x, y):
        logits = self(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.view(B * T)
        loss = nn.functional.cross_entropy(logits, y)
        return loss
