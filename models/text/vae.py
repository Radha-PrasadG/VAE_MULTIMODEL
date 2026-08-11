import torch
import torch.nn as nn

from .encoder import TextEncoder
from .decoder import TextDecoder


class TextVAE(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=64
    ):
        super().__init__()

        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        return mu + epsilon * std

    def forward(self, x):

        # Encoder
        mu, logvar = self.encoder(
            x
        )

        # Sampling
        z = self.reparameterize(
            mu,
            logvar
        )

        # Decoder
        reconstruction = self.decoder(
            x,
            z
        )

        return (
            reconstruction,
            mu,
            logvar
        )