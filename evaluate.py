import os
import math
import torch
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as ssim

from models.vae import VAE

# ==========================================
# Configuration
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 128

IMAGE_SIZE = 128

BATCH_SIZE = 1

MODEL_PATH = "checkpoints/best_vae.pth"

DATASET_PATH = "dataset/bottle/test"

OUTPUT_DIR = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Dataset
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

dataset = ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test Images :", len(dataset))

# ==========================================
# Load Model
# ==========================================

model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

print("Model Loaded Successfully")

# ==========================================
# Metrics
# ==========================================

total_mse = 0.0
total_psnr = 0.0
total_ssim = 0.0

count = 0

# ==========================================
# Evaluation
# ==========================================

with torch.no_grad():

    for images, _ in loader:

        images = images.to(DEVICE)

        reconstruction, _, _ = model(images)

        # -----------------------------
        # MSE
        # -----------------------------
        mse = F.mse_loss(
            reconstruction,
            images
        ).item()

        # -----------------------------
        # PSNR
        # -----------------------------
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * math.log10(1.0 / math.sqrt(mse))

        # -----------------------------
        # SSIM
        # -----------------------------
        original = images.squeeze().permute(1, 2, 0).cpu().numpy()

        reconstructed = (
            reconstruction.squeeze()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        score = ssim(
            original,
            reconstructed,
            channel_axis=2,
            data_range=1.0
        )

        total_mse += mse
        total_psnr += psnr
        total_ssim += score

        # -----------------------------------
        # Save comparison images
        # -----------------------------------

        comparison = torch.cat(
            [images.cpu(), reconstruction.cpu()],
            dim=0
        )

        save_image(
            comparison,
            os.path.join(
                OUTPUT_DIR,
                f"comparison_{count}.png"
            ),
            nrow=2,
            normalize=True
        )

        count += 1

# ==========================================
# Average Results
# ==========================================

avg_mse = total_mse / count
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count

print("\n==============================")
print("VAE Evaluation Results")
print("==============================")

print(f"Average MSE  : {avg_mse:.6f}")
print(f"Average PSNR : {avg_psnr:.2f} dB")
print(f"Average SSIM : {avg_ssim:.4f}")

# ==========================================
# Save Metrics
# ==========================================

with open(
    os.path.join(OUTPUT_DIR, "metrics.txt"),
    "w"
) as f:

    f.write("========== VAE Evaluation ==========\n\n")

    f.write(f"Average MSE  : {avg_mse:.6f}\n")
    f.write(f"Average PSNR : {avg_psnr:.2f} dB\n")
    f.write(f"Average SSIM : {avg_ssim:.4f}\n")

print("\nResults saved in evaluation/")