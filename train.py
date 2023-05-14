import argparse

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from models import GRU, LSTM, RNN, Bigram, Transformer

logger = TensorBoardLogger("logs", name="language_models")

models = {
    "bigram": Bigram,
    "rnn": RNN,
    "lstm": LSTM,
    "gru": GRU,
    "transformer": Transformer,
}


def create_tokenizer(corpus):
    tokens = set(corpus)
    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    idx_to_token = {idx: token for idx, token in enumerate(tokens)}
    return token_to_idx, idx_to_token


def encode_corpus(corpus, token_to_idx):
    return torch.tensor(
        [token_to_idx[token] for token in corpus], dtype=torch.long
    )


def split_data(data, val_size):
    split_idx = int(len(data) * (1 - val_size))
    return data[:split_idx], data[split_idx:]


class ShakespeareDataset(Dataset):
    def __init__(self, data, context=8):
        self.data = data
        self.context = context

    def __len__(self):
        return len(self.data) - self.context

    def __getitem__(self, idx):
        return (
            self.data[idx : idx + self.context],
            self.data[idx + 1 : idx + self.context + 1],
        )


if __name__ == "__main__":
    # Parse command line arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="rnn")
    args.add_argument("--context", type=int, default=16)
    args.add_argument("--batch_size", type=int, default=512)
    args.add_argument("--max_epochs", type=int, default=5)
    args.add_argument("--learning_rate", type=float, default=1e-3)
    args.add_argument("--val_size", type=float, default=0.1)
    args.add_argument("--corpus", type=str, default="data/shakespeare.txt")
    args = args.parse_args()

    # Read corpus from file
    with open(args.corpus) as file:
        corpus = file.read()

    # Create vocabulary index mapping
    token_to_idx, idx_to_token = create_tokenizer(corpus)
    vocab_size = len(token_to_idx)

    # Encode corpus
    data = encode_corpus(corpus, token_to_idx)

    # Split data into train and validation set
    train_data, val_data = split_data(data, args.val_size)
    train_loader = DataLoader(
        dataset=ShakespeareDataset(train_data, args.context),
        batch_size=args.batch_size,
    )
    val_loader = DataLoader(
        dataset=ShakespeareDataset(val_data, args.context),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Train model
    model = models[args.model](
        vocab_size=vocab_size,
        context=args.context,
        learning_rate=args.learning_rate,
        tokenizer=(token_to_idx, idx_to_token),
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[EarlyStopping("val_loss")],
        logger=logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save model checkpoint
    trainer.save_checkpoint(f"checkpoints/{args.model}.ckpt")
