import torch

from decoder import TabularDecoder


# Create sample latent vectors
z = torch.randn(4, 8)

# Create decoder
decoder = TabularDecoder(
    latent_dim=8,
    output_dim=16
)

# Forward pass
output = decoder(z)


print("=" * 50)
print("TABULAR VAE DECODER TEST")
print("=" * 50)

print("Latent input shape :", z.shape)
print("Output shape       :", output.shape)

print("\nMinimum output value:", output.min().item())
print("Maximum output value:", output.max().item())

print("\nDecoder test successful!")