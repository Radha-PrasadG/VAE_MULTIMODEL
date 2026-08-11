import torch
from encoder import Encoder

model = Encoder(latent_dim=128)

x = torch.randn(4, 3, 128, 128)

mu, log_var = model(x)

print("mu shape:", mu.shape)
print("log_var shape:", log_var.shape)