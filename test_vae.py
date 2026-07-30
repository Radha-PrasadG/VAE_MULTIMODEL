import torch
from models.vae import VAE

model = VAE(latent_dim=128)

x = torch.randn(4, 3, 128, 128)

reconstruction, mu, log_var = model(x)

print(reconstruction.shape)
print(mu.shape)
print(log_var.shape)