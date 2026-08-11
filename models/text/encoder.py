import torch
import torch.nn as nn


class TextEncoder(nn.Module):

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

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc_mu = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            hidden_dim,
            latent_dim
        )

    def forward(self, x):

        # Token IDs → embeddings
        embedded = self.embedding(x)

        # LSTM
        _, (hidden, _) = self.lstm(
            embedded
        )

        # Last hidden state
        hidden = hidden[-1]

        # Mean
        mu = self.fc_mu(
            hidden
        )

        # Log variance
        logvar = self.fc_logvar(
            hidden
        )

        return mu, logvar