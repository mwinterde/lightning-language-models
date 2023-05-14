import argparse
import time

import torch

from models import GRU, LSTM, RNN, Bigram, Transformer

models = {
    "bigram": Bigram,
    "rnn": RNN,
    "lstm": LSTM,
    "gru": GRU,
    "transformer": Transformer,
}


def generate_text(model, max_len=10000, delay=0.025):
    """
    Generate text from a trained model.
    """

    # Unpack tokenizer
    token_to_idx, idx_to_token = model.tokenizer

    # Initialize input
    input = torch.tensor([token_to_idx["\n"]], dtype=torch.long).view(1, 1)

    for _ in range(max_len):
        # Generate next token
        logits = model(input)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = torch.multinomial(probs, num_samples=1)

        # Print next token to stdout
        print(idx_to_token[output.item()], end="", flush=True)

        # Update input
        input = torch.cat([input[:, -model.context + 1 :], output], dim=-1)

        # Sleep to slow down generation
        time.sleep(delay)


if __name__ == "__main__":
    # Parse command line arguments
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="rnn")
    args = args.parse_args()

    # Load model
    model = models[args.model].load_from_checkpoint(
        f"checkpoints/{args.model}.ckpt"
    )

    # Generate text
    generate_text(model)
