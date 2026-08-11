import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder

    Input:
        Image (B,3,128,128)

    Output:
        reconstruction
        mu
        log_var
    """

    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization Trick

        z = mu + sigma * epsilon
        """

        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)

        z = mu + eps * std

        return z

    def forward(self, x):

        mu, log_var = self.encoder(x)

        z = self.reparameterize(mu, log_var)

        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var