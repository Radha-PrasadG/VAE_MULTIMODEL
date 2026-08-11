import torch
import torch.nn as nn

from encoder import TabularEncoder
from decoder import TabularDecoder


class TabularVAE(nn.Module):

    def __init__(self, input_dim=16, latent_dim=8):

        super(TabularVAE, self).__init__()

        self.encoder = TabularEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim
        )

        self.decoder = TabularDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim
        )

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        epsilon = torch.randn_like(std)

        z = mu + epsilon * std

        return z

    def forward(self, x):

        # Encoder
        mu, logvar = self.encoder(x)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decoder
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar