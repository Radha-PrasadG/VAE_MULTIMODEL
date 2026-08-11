import torch
import torch.nn as nn


class TabularEncoder(nn.Module):

    def __init__(self, input_dim=16, latent_dim=8):

        super(TabularEncoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Mean of latent distribution
        self.fc_mu = nn.Linear(32, latent_dim)

        # Log variance of latent distribution
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):

        x = self.encoder(x)

        mu = self.fc_mu(x)

        logvar = self.fc_logvar(x)

        return mu, logvar