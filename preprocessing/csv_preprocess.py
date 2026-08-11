import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../dataset/customer_churn/customer_churn_dataset-training-master.csv"

OUTPUT_DIR = "../dataset/customer_churn/processed"

# For the draft model, use 50,000 rows.
# Later, you can change this to None to use the complete dataset.
MAX_ROWS = None 


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("CUSTOMER CHURN VAE PREPROCESSING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

print(df.isnull().sum())

df = df.dropna().reset_index(drop=True)

print("\nAfter removing missing rows:")
print(df.shape)


# ============================================================
# SAMPLE DATA FOR DRAFT TRAINING
# ============================================================

if MAX_ROWS is not None and len(df) > MAX_ROWS:

    df = df.sample(
        n=MAX_ROWS,
        random_state=42
    ).reset_index(drop=True)

    print("\nUsing sample for draft training:")
    print(df.shape)

else:

    print("\nUsing complete dataset:")
    print(df.shape)


# ============================================================
# REMOVE CUSTOMER ID
# ============================================================

if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

print("\nCustomerID removed.")


# ============================================================
# DEFINE COLUMNS
# ============================================================

numerical_columns = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Churn"
]

categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]


# ============================================================
# DISPLAY COLUMN INFORMATION
# ============================================================

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Churn"]
)

print("\nTraining data:", train_df.shape)
print("Testing data :", test_df.shape)


# ============================================================
# NUMERICAL PREPROCESSING
# ============================================================

scaler = MinMaxScaler()

train_numeric = scaler.fit_transform(
    train_df[numerical_columns]
)

test_numeric = scaler.transform(
    test_df[numerical_columns]
)


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

train_categorical = encoder.fit_transform(
    train_df[categorical_columns]
)

test_categorical = encoder.transform(
    test_df[categorical_columns]
)


# ============================================================
# COMBINE NUMERICAL + CATEGORICAL FEATURES
# ============================================================

X_train = np.concatenate(
    [
        train_numeric,
        train_categorical
    ],
    axis=1
)

X_test = np.concatenate(
    [
        test_numeric,
        test_categorical
    ],
    axis=1
)


# ============================================================
# DISPLAY FINAL SHAPE
# ============================================================

print("\nFinal processed training shape:")
print(X_train.shape)

print("\nFinal processed testing shape:")
print(X_test.shape)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train.astype(np.float32)
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test.astype(np.float32)
)


# ============================================================
# SAVE PREPROCESSING OBJECTS
# ============================================================

joblib.dump(
    scaler,
    os.path.join(OUTPUT_DIR, "scaler.pkl")
)

joblib.dump(
    encoder,
    os.path.join(OUTPUT_DIR, "encoder.pkl")
)


# ============================================================
# SAVE COLUMN INFORMATION
# ============================================================

metadata = {
    "numerical_columns": numerical_columns,
    "categorical_columns": categorical_columns,
    "input_dimension": X_train.shape[1]
}

joblib.dump(
    metadata,
    os.path.join(OUTPUT_DIR, "metadata.pkl")
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles created:")

print("X_train.npy")
print("X_test.npy")
print("scaler.pkl")
print("encoder.pkl")
print("metadata.pkl")

print("\nFinal input dimension:", X_train.shape[1])