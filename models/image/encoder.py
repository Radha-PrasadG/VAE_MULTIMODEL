import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional Encoder for Variational Autoencoder (VAE)

    Input:
        (B, 3, 128, 128)

    Output:
        mu      -> (B, latent_dim)
        log_var -> (B, latent_dim)
    """

    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16 -> 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8 -> 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Flatten size = 512 x 4 x 4 = 8192
        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):

        x = self.features(x)

        x = self.flatten(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var