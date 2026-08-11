import torch

from vae import TabularVAE


# Create sample input
x = torch.rand(4, 16)


# Create VAE
model = TabularVAE(
    input_dim=16,
    latent_dim=8
)


# Forward pass
reconstruction, mu, logvar = model(x)


print("=" * 60)
print("TABULAR VAE TEST")
print("=" * 60)

print("Input shape          :", x.shape)
print("Reconstruction shape :", reconstruction.shape)
print("Mean shape           :", mu.shape)
print("Logvar shape         :", logvar.shape)

print("\nLatent dimension     :", mu.shape[1])

print("\nVAE test successful!")