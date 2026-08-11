import torch
import torch.nn.functional as F


def text_vae_loss(
    reconstruction,
    target,
    mu,
    logvar
):

    # Reconstruction loss
    reconstruction_loss = F.cross_entropy(
        reconstruction.reshape(
            -1,
            reconstruction.size(-1)
        ),
        target.reshape(-1),
        ignore_index=0
    )

    # KL divergence
    kl_loss = -0.5 * torch.mean(
        1 + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    # Total VAE loss
    loss = (
        reconstruction_loss
        + 0.001 * kl_loss
    )

    return (
        loss,
        reconstruction_loss,
        kl_loss
    )