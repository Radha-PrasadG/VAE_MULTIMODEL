import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from models.vae import VAE

# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = 128

BATCH_SIZE = 32

LATENT_DIM = 128

EPOCHS = 50

LEARNING_RATE = 1e-3

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)

# ==========================================================
# Dataset
# ==========================================================

DATASET_PATH = "dataset/bottle/train"

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(5),

    transforms.ToTensor()
])

dataset = ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(f"Training Images : {len(dataset)}")

# ==========================================================
# Directories
# ==========================================================

CHECKPOINT_DIR = "checkpoints"

OUTPUT_DIR = "outputs"

LOG_DIR = "logs"

PLOT_DIR = "plots"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)

os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================================
# CSV Logger
# ==========================================================

csv_file = os.path.join(
    LOG_DIR,
    "training_log.csv"
)

with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Epoch",
        "Total Loss",
        "Reconstruction Loss",
        "KL Loss"
    ])

print("Training log initialized.")

# ==========================================================
# Model
# ==========================================================

model = VAE(
    latent_dim=LATENT_DIM
).to(DEVICE)

print("\nModel Initialized Successfully!")

# ==========================================================
# Optimizer
# ==========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

print("Optimizer : Adam")
print("Scheduler : ReduceLROnPlateau")

# ==========================================================
# Loss Function
# ==========================================================

reconstruction_loss_fn = nn.MSELoss(
    reduction="sum"
)


def vae_loss(reconstruction, original, mu, log_var):
    """
    Computes the VAE loss.

    Total Loss = Reconstruction Loss + KL Divergence
    """

    reconstruction_loss = reconstruction_loss_fn(
        reconstruction,
        original
    )

    kl_loss = -0.5 * torch.sum(
        1 +
        log_var -
        mu.pow(2) -
        log_var.exp()
    )

    total_loss = reconstruction_loss + kl_loss

    return (
        total_loss,
        reconstruction_loss,
        kl_loss
    )

# ==========================================================
# Best Model Saving
# ==========================================================

best_loss = float("inf")

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_vae.pth"
)

# ==========================================================
# Early Stopping
# ==========================================================

EARLY_STOPPING_PATIENCE = 10

early_stop_counter = 0

# ==========================================================
# Training History
# ==========================================================

loss_history = []

reconstruction_history = []

kl_history = []

learning_rate_history = []

print("\nTraining configuration completed.")

print("=" * 60)
print(f"Epochs        : {EPOCHS}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Learning Rate : {LEARNING_RATE}")
print(f"Latent Dim    : {LATENT_DIM}")
print("=" * 60)

# ==========================================================
# Training Loop
# ==========================================================

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    running_reconstruction = 0.0
    running_kl = 0.0

    for images, _ in train_loader:

        images = images.to(DEVICE)

        optimizer.zero_grad()

        # -----------------------------
        # Forward Pass
        # -----------------------------
        reconstruction, mu, log_var = model(images)

        # -----------------------------
        # Compute Loss
        # -----------------------------
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            log_var
        )

        # -----------------------------
        # Backpropagation
        # -----------------------------
        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item()

        running_reconstruction += reconstruction_loss.item()

        running_kl += kl_loss.item()

    # ======================================================
    # Epoch Statistics
    # ======================================================

    avg_loss = running_loss / len(dataset)

    avg_reconstruction = running_reconstruction / len(dataset)

    avg_kl = running_kl / len(dataset)

    current_lr = optimizer.param_groups[0]["lr"]

    loss_history.append(avg_loss)

    reconstruction_history.append(avg_reconstruction)

    kl_history.append(avg_kl)

    learning_rate_history.append(current_lr)

    # ======================================================
    # Scheduler
    # ======================================================

    scheduler.step(avg_loss)

    # ======================================================
    # Save CSV Log
    # ======================================================

    with open(csv_file, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            epoch + 1,
            avg_loss,
            avg_reconstruction,
            avg_kl
        ])

    # ======================================================
    # Console Output
    # ======================================================

    print(
        f"Epoch [{epoch+1:03d}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Recon: {avg_reconstruction:.4f} | "
        f"KL: {avg_kl:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    # ======================================================
    # Save Reconstruction Image
    # ======================================================

    if (epoch + 1) % 5 == 0:

        comparison = torch.cat(
            [
                images[:8].cpu(),
                reconstruction[:8].cpu()
            ],
            dim=0
        )

        save_image(
            comparison,
            os.path.join(
                OUTPUT_DIR,
                f"epoch_{epoch+1}.png"
            ),
            nrow=8,
            normalize=True
        )

    # ======================================================
    # Save Best Model
    # ======================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        early_stop_counter = 0

        print("Best model updated.")

    else:

        early_stop_counter += 1

    # ======================================================
    # Save Checkpoint Every 10 Epochs
    # ======================================================

    if (epoch + 1) % 10 == 0:

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"vae_epoch_{epoch+1}.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

    # ======================================================
    # Early Stopping
    # ======================================================

    if early_stop_counter >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered!")

        break

print("\nTraining Finished Successfully!")

# ==========================================================
# Training Completed
# ==========================================================

print("\n" + "=" * 60)
print("Training Completed Successfully")
print("=" * 60)

# ==========================================================
# Plot Training Loss
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    loss_history,
    label="Total Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "loss_curve.png"
    )
)

plt.close()

# ==========================================================
# Plot Reconstruction Loss
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    reconstruction_history,
    label="Reconstruction Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Reconstruction Loss")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "reconstruction_loss.png"
    )
)

plt.close()

# ==========================================================
# Plot KL Divergence
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    kl_history,
    label="KL Divergence",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("KL Divergence")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "kl_loss.png"
    )
)

plt.close()

# ==========================================================
# Plot Learning Rate
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    learning_rate_history,
    label="Learning Rate",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "learning_rate.png"
    )
)

plt.close()

# ==========================================================
# Save Final Metrics
# ==========================================================

summary_file = os.path.join(
    LOG_DIR,
    "training_summary.txt"
)

with open(summary_file, "w") as file:

    file.write("========== VAE TRAINING SUMMARY ==========\n\n")

    file.write(f"Epochs Trained       : {len(loss_history)}\n")
    file.write(f"Batch Size           : {BATCH_SIZE}\n")
    file.write(f"Learning Rate        : {LEARNING_RATE}\n")
    file.write(f"Latent Dimension     : {LATENT_DIM}\n")
    file.write(f"Best Loss            : {best_loss:.6f}\n")
    file.write(f"Training Images      : {len(dataset)}\n")
    file.write(f"Device               : {DEVICE}\n")

print("\nTraining summary saved.")

# ==========================================================
# Final Console Summary
# ==========================================================

print("\n" + "=" * 60)
print("FINAL TRAINING REPORT")
print("=" * 60)

print(f"Training Images      : {len(dataset)}")
print(f"Epochs Completed     : {len(loss_history)}")
print(f"Best Loss            : {best_loss:.6f}")
print(f"Device               : {DEVICE}")

print("\nGenerated Files")

print("------------------------------")

print("✓ checkpoints/best_vae.pth")

print("✓ logs/training_log.csv")

print("✓ logs/training_summary.txt")

print("✓ plots/loss_curve.png")

print("✓ plots/reconstruction_loss.png")

print("✓ plots/kl_loss.png")

print("✓ plots/learning_rate.png")

print("✓ outputs/epoch_x.png")

print("\nResearch-grade VAE Training Completed!")