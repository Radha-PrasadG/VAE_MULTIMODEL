import os
import torch

from torchinfo import summary

from models.vae import VAE

# ==========================================
# Count Images in a Folder
# ==========================================

def count_images(folder_path):
    if not os.path.exists(folder_path):
        return 0

    count = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                count += 1

    return count
# ==========================================
# Configuration
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LATENT_DIM = 128

OUTPUT_DIR = "model_info"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
# ==========================================
# Dataset Paths
# ==========================================

TRAIN_DATASET = "dataset/bottle/train"
TEST_DATASET = "dataset/bottle/test"
SYNTHETIC_DATASET = "generated_dataset"

# ==========================================
# Load Model
# ==========================================

model = VAE(
    latent_dim=LATENT_DIM
).to(DEVICE)


model.eval()


print("=" * 60)
print("VAE MODEL SUMMARY")
print("=" * 60)


# ==========================================
# Generate Model Summary
# ==========================================

model_summary = summary(
    model,
    input_size=(1, 3, 128, 128),
    verbose=0
)


# Print to console

print(model_summary)


# ==========================================
# Parameter Calculation
# ==========================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


encoder_params = sum(
    p.numel()
    for p in model.encoder.parameters()
)


decoder_params = sum(
    p.numel()
    for p in model.decoder.parameters()
)

# ==========================================
# Dataset Statistics
# ==========================================

train_images = count_images(TRAIN_DATASET)

test_images = count_images(TEST_DATASET)

synthetic_images = count_images(SYNTHETIC_DATASET)

total_original = train_images + test_images

# ==========================================
# Save Summary
# ==========================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "model_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "==============================\n"
    )

    file.write(
        "VAE MODEL SUMMARY\n"
    )

    file.write(
        "==============================\n\n"
    )


    # torchinfo summary

    file.write(
        str(model_summary)
    )


    file.write(
        "\n\n\n"
    )


    file.write(
        "==============================\n"
    )

    file.write(
        "PARAMETER INFORMATION\n"
    )

    file.write(
        "==============================\n\n"
    )


    file.write(
        f"Total Parameters      : {total_params:,}\n"
    )

    file.write(
        f"Trainable Parameters  : {trainable_params:,}\n"
    )

    file.write(
        f"Encoder Parameters    : {encoder_params:,}\n"
    )

    file.write(
        f"Decoder Parameters    : {decoder_params:,}\n"
    )

    file.write(
        f"Latent Dimension      : {LATENT_DIM}\n"
    )

    file.write(
        f"Input Resolution      : 128x128x3\n"
    )

    file.write(
        f"Device Used           : {DEVICE}\n"
    )
    file.write(
        "==============================\n\n"
    )
    file.write("DATASET INFORMATION\n")
    file.write(
        "==============================\n\n"
    )

    file.write(f"Training Images           : {train_images}\n")
    file.write(f"Testing Images            : {test_images}\n")
    file.write(f"Original Dataset Images   : {total_original}\n")
    file.write(f"Synthetic Images Created  : {synthetic_images}\n")

# ==========================================
# Console Report
# ==========================================

print("\n")
print("=" * 60)
print("PARAMETER INFORMATION")
print("=" * 60)

print(
    f"Total Parameters      : {total_params:,}"
)

print(
    f"Trainable Parameters  : {trainable_params:,}"
)

print(
    f"Encoder Parameters    : {encoder_params:,}"
)

print(
    f"Decoder Parameters    : {decoder_params:,}"
)

print(
    f"Latent Dimension      : {LATENT_DIM}"
)

print(
    f"Device Used           : {DEVICE}"
)

print("\nDataset Information")
print("------------------------------")

print(f"Training Images       : {train_images}")
print(f"Testing Images        : {test_images}")
print(f"Original Dataset      : {total_original}")
print(f"Synthetic Dataset     : {synthetic_images}")
print("\nModel summary saved successfully!")

print(
    f"Location: {summary_path}"
)