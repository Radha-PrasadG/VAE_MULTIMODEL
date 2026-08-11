import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ============================================================
# ALLOW IMPORTING FROM models/csv
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "csv"
        )
    )
)

from vae import TabularVAE
from loss import vae_loss


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "../dataset/customer_churn/processed/X_train.npy"

CHECKPOINT_DIR = "../checkpoints"

PLOTS_DIR = "../plots/CSV VAE"


# Create directories if they do not exist
os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 256

EPOCHS = 30

LEARNING_RATE = 0.001

INPUT_DIM = 16

LATENT_DIM = 8


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)

print("CUSTOMER CHURN TABULAR VAE TRAINING")

print("=" * 60)

print("\nDevice:", device)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

if not os.path.exists(TRAIN_PATH):

    raise FileNotFoundError(
        f"Training data not found:\n{TRAIN_PATH}"
    )


X_train = np.load(
    TRAIN_PATH
)

print(
    "Training data shape:",
    X_train.shape
)


# ============================================================
# CONVERT NUMPY → PYTORCH TENSOR
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)


# ============================================================
# DATA LOADER
# ============================================================

dataset = TensorDataset(
    X_train
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# CREATE MODEL
# ============================================================

model = TabularVAE(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM
)

model = model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

loss_history = []

reconstruction_history = []

kl_history = []


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    total_reconstruction = 0.0

    total_kl = 0.0


    # --------------------------------------------------------
    # PROCESS EACH BATCH
    # --------------------------------------------------------

    for (batch,) in dataloader:

        batch = batch.to(device)


        # ----------------------------------------------------
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        reconstruction, mu, logvar = model(
            batch
        )


        # ----------------------------------------------------
        # CALCULATE VAE LOSS
        # ----------------------------------------------------

        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction,
            batch,
            mu,
            logvar
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # UPDATE MODEL
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # ACCUMULATE LOSSES
        # ----------------------------------------------------

        total_loss += loss.item()

        total_reconstruction += (
            reconstruction_loss.item()
        )

        total_kl += (
            kl_loss.item()
        )


    # ========================================================
    # AVERAGE LOSSES
    # ========================================================

    num_batches = len(dataloader)

    avg_loss = (
        total_loss /
        num_batches
    )

    avg_reconstruction = (
        total_reconstruction /
        num_batches
    )

    avg_kl = (
        total_kl /
        num_batches
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    loss_history.append(
        avg_loss
    )

    reconstruction_history.append(
        avg_reconstruction
    )

    kl_history.append(
        avg_kl
    )


    # ========================================================
    # DISPLAY TRAINING PROGRESS
    # ========================================================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.6f} | "
        f"Reconstruction: {avg_reconstruction:.6f} | "
        f"KL: {avg_kl:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    CHECKPOINT_DIR,
    "tabular_vae.pth"
)


torch.save(
    model.state_dict(),
    model_path
)


print("\n" + "=" * 60)

print("TRAINING COMPLETED")

print("=" * 60)

print("\nModel saved to:")

print(model_path)


# ============================================================
# GENERATE PLOTS
# ============================================================

print("\nGenerating training plots...")


epochs = range(
    1,
    EPOCHS + 1
)


# ============================================================
# 1. TOTAL LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    loss_history,
    linewidth=2
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Total Loss"
)

plt.title(
    "Customer Churn VAE - Training Loss"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


loss_plot_path = os.path.join(
    PLOTS_DIR,
    "loss_curve.png"
)


plt.savefig(
    loss_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 2. RECONSTRUCTION LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    reconstruction_history,
    linewidth=2
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Reconstruction Loss"
)

plt.title(
    "Customer Churn VAE - Reconstruction Loss"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


reconstruction_plot_path = os.path.join(
    PLOTS_DIR,
    "reconstruction_loss.png"
)


plt.savefig(
    reconstruction_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 3. KL DIVERGENCE LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    kl_history,
    linewidth=2
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "KL Divergence"
)

plt.title(
    "Customer Churn VAE - KL Divergence Loss"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


kl_plot_path = os.path.join(
    PLOTS_DIR,
    "kl_loss.png"
)


plt.savefig(
    kl_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    PLOTS_DIR,
    "training_history.csv"
)


history = np.column_stack(
    (
        list(epochs),
        loss_history,
        reconstruction_history,
        kl_history
    )
)


np.savetxt(
    history_path,
    history,
    delimiter=",",
    header=(
        "epoch,"
        "total_loss,"
        "reconstruction_loss,"
        "kl_loss"
    ),
    comments=""
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print("TRAINING PLOTS GENERATED SUCCESSFULLY")

print("=" * 60)

print("\nPlot directory:")

print(
    os.path.abspath(
        PLOTS_DIR
    )
)

print("\nGenerated files:")

print("1. loss_curve.png")

print("2. reconstruction_loss.png")

print("3. kl_loss.png")

print("4. training_history.csv")

print("\nModel checkpoint:")

print("5. tabular_vae.pth")

print("\nAll tasks completed successfully!")