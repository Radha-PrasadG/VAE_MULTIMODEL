import os
import torch

from torchvision.utils import save_image

from models.vae import VAE

# ==========================================
# Configuration
# ==========================================

LATENT_DIM = 128

NUM_IMAGES = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/best_vae.pth"

OUTPUT_DIR = "generated_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Load Model
# ==========================================

model = VAE(
    latent_dim=LATENT_DIM
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Best model loaded.")

# ==========================================
# Generate Images
# ==========================================

with torch.no_grad():

    for i in range(NUM_IMAGES):

        z = torch.randn(
            1,
            LATENT_DIM
        ).to(DEVICE)

        image = model.decoder(z)

        save_image(
            image,
            os.path.join(
                OUTPUT_DIR,
                f"{i+1:04d}.png"
            ),
            normalize=True
        )

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{NUM_IMAGES} images generated")

print("\nSynthetic dataset created successfully!")