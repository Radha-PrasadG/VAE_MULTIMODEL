import os
import math

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from PIL import Image

from torchvision import transforms

from skimage.metrics import structural_similarity as ssim

from models.vae import VAE

# ===========================================
# Configuration
# ===========================================

IMAGE_SIZE = 128

LATENT_DIM = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/best_vae.pth"

# ===========================================
# Automatically Select a Test Image
# ===========================================

TEST_FOLDER = "dataset/bottle/test/good"

image_list = [
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]

if len(image_list) == 0:
    raise FileNotFoundError(
        f"No images found in {TEST_FOLDER}"
    )

IMAGE_PATH = os.path.join(
    TEST_FOLDER,
    image_list[0]
)

print(f"Using image: {IMAGE_PATH}")

OUTPUT_DIR = "predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================
# Load Model
# ===========================================

model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Best model loaded.")

# ===========================================
# Image Transform
# ===========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# ===========================================
# Load Image
# ===========================================

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ===========================================
# Prediction
# ===========================================

with torch.no_grad():

    reconstruction, _, _ = model(image_tensor)

# ===========================================
# Metrics
# ===========================================

mse = F.mse_loss(
    reconstruction,
    image_tensor
).item()

if mse == 0:
    psnr = 100
else:
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))

original = image_tensor.squeeze().permute(1,2,0).cpu().numpy()

reconstructed = reconstruction.squeeze().permute(1,2,0).cpu().numpy()

ssim_score = ssim(
    original,
    reconstructed,
    channel_axis=2,
    data_range=1.0
)

# ===========================================
# Difference Heatmap
# ===========================================

difference = abs(original - reconstructed)

# ===========================================
# Save Images
# ===========================================

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(reconstructed)
plt.title("Reconstruction")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(difference)
plt.title("Difference")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "prediction_result.png"
    )
)

plt.close()

# ===========================================
# Save Metrics
# ===========================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "prediction_metrics.txt"
    ),
    "w"
) as file:

    file.write("===== Prediction Results =====\n\n")

    file.write(f"MSE  : {mse:.6f}\n")
    file.write(f"PSNR : {psnr:.2f} dB\n")
    file.write(f"SSIM : {ssim_score:.4f}\n")

print("\nPrediction Complete")

print(f"MSE  : {mse:.6f}")
print(f"PSNR : {psnr:.2f}")
print(f"SSIM : {ssim_score:.4f}")

print("\nResults saved in predictions/")