import os
import sys

import torch
import pandas as pd
import numpy as np
import joblib


# ============================================================
# IMPORT MODEL
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


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "../checkpoints/tabular_vae.pth"

PROCESSED_DIR = "../dataset/customer_churn/processed"

OUTPUT_DIR = "../generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 16
LATENT_DIM = 8

NUM_SAMPLES = 1000


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("CUSTOMER CHURN SYNTHETIC DATA GENERATION")
print("=" * 60)

print("\nLoading trained VAE...")

model = TabularVAE(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# GENERATE LATENT VECTORS
# ============================================================

print("\nGenerating latent vectors...")

z = torch.randn(
    NUM_SAMPLES,
    LATENT_DIM
)


# ============================================================
# DECODE
# ============================================================

with torch.no_grad():

    generated = model.decoder(z)


generated = generated.numpy()

print("Generated array shape:", generated.shape)


# ============================================================
# LOAD PREPROCESSING OBJECTS
# ============================================================

scaler = joblib.load(
    os.path.join(
        PROCESSED_DIR,
        "scaler.pkl"
    )
)

encoder = joblib.load(
    os.path.join(
        PROCESSED_DIR,
        "encoder.pkl"
    )
)

metadata = joblib.load(
    os.path.join(
        PROCESSED_DIR,
        "metadata.pkl"
    )
)

# ============================================================
# SPLIT NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

num_columns = metadata["numerical_columns"]
cat_columns = metadata["categorical_columns"]

num_features = len(num_columns)

generated_numeric = generated[:, :num_features]

generated_categorical = generated[:, num_features:]


# ============================================================
# REVERSE NUMERICAL SCALING
# ============================================================

generated_numeric = scaler.inverse_transform(
    generated_numeric
)


# ============================================================
# RECONSTRUCT CATEGORICAL FEATURES
# ============================================================

category_sizes = [
    len(categories)
    for categories in encoder.categories_
]

decoded_categories = []

start = 0

for size, column in zip(
    category_sizes,
    cat_columns
):

    end = start + size

    values = generated_categorical[:, start:end]

    indices = np.argmax(
        values,
        axis=1
    )

    categories = encoder.categories_[
        len(decoded_categories)
    ]

    decoded = categories[indices]

    decoded_categories.append(decoded)

    start = end


# ============================================================
# CREATE DATAFRAME
# ============================================================

synthetic_df = pd.DataFrame(
    generated_numeric,
    columns=num_columns
)


for column, values in zip(
    cat_columns,
    decoded_categories
):

    synthetic_df[column] = values


# ============================================================
# REORDER COLUMNS
# ============================================================

synthetic_df = synthetic_df[
    num_columns + cat_columns
]


# ============================================================
# SAVE CSV
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "synthetic_customer_churn.csv"
)

synthetic_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GENERATION COMPLETED")
print("=" * 60)

print("\nSynthetic records:", len(synthetic_df))

print("\nGenerated columns:")
print(synthetic_df.columns.tolist())

print("\nFirst 5 synthetic records:")
print(synthetic_df.head())

print("\nSaved to:")
print(output_path)