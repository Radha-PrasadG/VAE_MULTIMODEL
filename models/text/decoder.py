import torch
import torch.nn as nn


class TextDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=64
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.latent_to_hidden = nn.Linear(
            latent_dim,
            hidden_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(
        self,
        x,
        z
    ):

        # Convert latent vector
        # into initial hidden state
        hidden = torch.tanh(
            self.latent_to_hidden(z)
        )

        hidden = hidden.unsqueeze(0)

        # Initial cell state
        cell = torch.zeros_like(
            hidden
        )

        # Token embedding
        embedded = self.embedding(x)

        # LSTM
        output, _ = self.lstm(
            embedded,
            (hidden, cell)
        )

        # Vocabulary probabilities/logits
        logits = self.output_layer(
            output
        )

        return logits