import pandas as pd

file_path = "../dataset/customer_churn/customer_churn_dataset-training-master.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CUSTOMER CHURN DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumerical Columns:")
print(df.select_dtypes(include=["number"]).columns.tolist())

print("\nCategorical Columns:")
print(df.select_dtypes(include=["object"]).columns.tolist())