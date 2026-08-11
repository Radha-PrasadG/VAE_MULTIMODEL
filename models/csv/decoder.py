import torch
import torch.nn as nn


class TabularDecoder(nn.Module):

    def __init__(self, latent_dim=8, output_dim=16):

        super(TabularDecoder, self).__init__()

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim),

            # Our numerical preprocessing is in the range [0, 1]
            nn.Sigmoid()
        )

    def forward(self, z):

        return self.decoder(z)