import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models.vae import VAE

# ==========================================
# Configuration
# ==========================================

IMAGE_SIZE = 128
LATENT_DIM = 128
STEPS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/best_vae.pth"

TEST_FOLDER = "dataset/bottle/test/good"

OUTPUT_DIR = "interpolation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Load Model
# ==========================================

model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model Loaded Successfully")

# ==========================================
# Image Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# ==========================================
# Load Two Images
# ==========================================

images = sorted([
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if len(images) < 2:
    raise ValueError("Need at least two images.")

img1 = Image.open(os.path.join(TEST_FOLDER, images[0])).convert("RGB")
img2 = Image.open(os.path.join(TEST_FOLDER, images[1])).convert("RGB")

img1 = transform(img1).unsqueeze(0).to(DEVICE)
img2 = transform(img2).unsqueeze(0).to(DEVICE)

# ==========================================
# Encode
# ==========================================

with torch.no_grad():

    _, mu1, _ = model(img1)
    _, mu2, _ = model(img2)

# ==========================================
# Interpolation
# ==========================================

generated = []

with torch.no_grad():

    for alpha in np.linspace(0, 1, STEPS):

        z = (1 - alpha) * mu1 + alpha * mu2

        output = model.decoder(z)

        generated.append(output.cpu())

generated = torch.cat(generated, dim=0)

save_image(
    generated,
    os.path.join(
        OUTPUT_DIR,
        "latent_interpolation.png"
    ),
    nrow=STEPS,
    normalize=True
)

print("Interpolation completed.")