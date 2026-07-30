import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Convolutional Decoder for Variational Autoencoder (VAE)

    Input:
        Latent vector (B, latent_dim)

    Output:
        Reconstructed image (B, 3, 128, 128)
    """

    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()

        # Project latent vector to feature map
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(

            # 4x4 -> 8x8
            nn.ConvTranspose2d(
                512, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(
                32, 3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Output pixels between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, 512, 4, 4)

        x = self.decoder(x)

        return x