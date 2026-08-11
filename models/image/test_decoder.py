import torch
from decoder import Decoder

model = Decoder(latent_dim=128)

z = torch.randn(4, 128)

output = model(z)

print(output.shape)