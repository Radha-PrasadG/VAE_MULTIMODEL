import pandas as pd
import os

PATH = "../dataset/sms/SMSSpamCollection"

print("=" * 60)
print("SMS SPAM COLLECTION DATASET")
print("=" * 60)

df = pd.read_csv(
    PATH,
    sep="\t",
    header=None,
    names=["label", "text"]
)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nData types:")
print(df.dtypes)