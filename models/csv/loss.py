import torch
import torch.nn.functional as F


def vae_loss(reconstruction, original, mu, logvar):

    # Reconstruction loss
    reconstruction_loss = F.mse_loss(
        reconstruction,
        original,
        reduction="mean"
    )

    # KL divergence
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    # Total VAE loss
    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss