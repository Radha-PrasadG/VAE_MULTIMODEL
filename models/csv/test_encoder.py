import torch

from encoder import TabularEncoder


# Create sample input
x = torch.randn(4, 16)

# Create encoder
encoder = TabularEncoder(
    input_dim=16,
    latent_dim=8
)

# Forward pass
mu, logvar = encoder(x)


print("=" * 50)
print("TABULAR VAE ENCODER TEST")
print("=" * 50)

print("Input shape     :", x.shape)
print("Mean shape      :", mu.shape)
print("Logvar shape    :", logvar.shape)

print("\nEncoder test successful!")