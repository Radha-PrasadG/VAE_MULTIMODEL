import os
import torch
from torchvision.utils import save_image

from models.vae import VAE

# ===========================
# Configuration
# ===========================

LATENT_DIM = 128
NUM_IMAGES = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/best_vae.pth"

OUTPUT_DIR = "generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# Load Model
# ===========================

model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

model.eval()

print("Model Loaded Successfully!")

# ===========================
# Generate Images
# ===========================

with torch.no_grad():

    # Sample latent vectors from N(0,1)
    z = torch.randn(NUM_IMAGES, LATENT_DIM).to(DEVICE)

    generated_images = model.decoder(z)

    save_image(
        generated_images,
        os.path.join(OUTPUT_DIR, "synthetic_bottles.png"),
        nrow=4,
        normalize=True
    )

print("Synthetic images saved successfully!")