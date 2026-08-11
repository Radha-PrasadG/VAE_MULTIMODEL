import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from models.text.vae import TextVAE
from models.text.loss import text_vae_loss


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = (
    "../dataset/sms/processed/X_train.npy"
)

METADATA_PATH = (
    "../dataset/sms/processed/metadata.pkl"
)

CHECKPOINT_DIR = "../checkpoints"

PLOTS_DIR = "../plots/TEXT VAE"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 0.001

EMBEDDING_DIM = 128

HIDDEN_DIM = 256

LATENT_DIM = 64


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)
print("SMS TEXT VAE TRAINING")
print("=" * 60)

print("\nDevice:", device)


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

metadata = joblib.load(
    METADATA_PATH
)

VOCAB_SIZE = metadata["vocab_size"]

MAX_LENGTH = metadata[
    "max_sequence_length"
]


print(
    "Vocabulary size:",
    VOCAB_SIZE
)

print(
    "Sequence length:",
    MAX_LENGTH
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

X_train = np.load(
    TRAIN_PATH
)

print(
    "Training data shape:",
    X_train.shape
)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.long
)


# ============================================================
# DATASET
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

model = TextVAE(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
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
# LOSS HISTORY
# ============================================================

loss_history = []

reconstruction_history = []

kl_history = []


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    total_reconstruction = 0

    total_kl = 0


    for (batch,) in dataloader:

        batch = batch.to(device)

        optimizer.zero_grad()


        # Forward pass
        reconstruction, mu, logvar = model(
            batch
        )


        # Calculate loss
        loss, reconstruction_loss, kl_loss = (
            text_vae_loss(
                reconstruction,
                batch,
                mu,
                logvar
            )
        )


        # Backpropagation
        loss.backward()


        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_loss += loss.item()

        total_reconstruction += (
            reconstruction_loss.item()
        )

        total_kl += kl_loss.item()


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


    loss_history.append(
        avg_loss
    )

    reconstruction_history.append(
        avg_reconstruction
    )

    kl_history.append(
        avg_kl
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.6f} | "
        f"Reconstruction: "
        f"{avg_reconstruction:.6f} | "
        f"KL: {avg_kl:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    CHECKPOINT_DIR,
    "text_vae.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# SAVE LOSS DATA
# ============================================================

np.save(
    os.path.join(
        PLOTS_DIR,
        "loss_history.npy"
    ),
    np.array(loss_history)
)

np.save(
    os.path.join(
        PLOTS_DIR,
        "reconstruction_history.npy"
    ),
    np.array(
        reconstruction_history
    )
)

np.save(
    os.path.join(
        PLOTS_DIR,
        "kl_history.npy"
    ),
    np.array(
        kl_history
    )
)


# ============================================================
# PLOT TOTAL LOSS
# ============================================================

plt.figure()

plt.plot(
    loss_history
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Text VAE Training Loss"
)

plt.grid()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "loss_curve.png"
    )
)

plt.close()


# ============================================================
# PLOT RECONSTRUCTION LOSS
# ============================================================

plt.figure()

plt.plot(
    reconstruction_history
)

plt.xlabel("Epoch")

plt.ylabel(
    "Reconstruction Loss"
)

plt.title(
    "Text VAE Reconstruction Loss"
)

plt.grid()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "reconstruction_loss.png"
    )
)

plt.close()


# ============================================================
# PLOT KL LOSS
# ============================================================

plt.figure()

plt.plot(
    kl_history
)

plt.xlabel("Epoch")

plt.ylabel("KL Divergence")

plt.title(
    "Text VAE KL Loss"
)

plt.grid()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "kl_loss.png"
    )
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)

print("TEXT VAE TRAINING COMPLETED")

print("=" * 60)

print("\nModel saved to:")

print(model_path)

print("\nPlots saved to:")

print(PLOTS_DIR)