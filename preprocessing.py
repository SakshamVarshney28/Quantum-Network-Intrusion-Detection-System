import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("PREPROCESSING DATASET")
print("=" * 60)

print("\nLoading Dataset...")

df = pd.read_csv("combined_dataset.csv")

print("Dataset Loaded")
print("Shape :", df.shape)

print("\nChecking Missing Values...")

print(df.isnull().sum())

print("\nRemoving Missing Values...")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna()

print("Done")

print("\nRemoving Duplicate Rows...")

df = df.drop_duplicates()

print("Done")

print("\nFinal Shape :", df.shape)
# --------------------------------------------------
# Separate Features and Labels
# --------------------------------------------------

print("\nSeparating Features and Labels...")

# "label" (0/1) is what the Hybrid QNN trains on.
# "attack_type" (DDoS, DoS, PortScan, ...) is NOT a model feature and
# is dropped from X here too — but we keep it alongside as its own
# target so a second-stage classifier can be trained on it later,
# using the exact same train/test split as the QNN.
has_attack_type = "attack_type" in df.columns

X = df.drop(columns=["label", "attack_type"] if has_attack_type else ["label"])
y = df["label"]
y_attack = df["attack_type"] if has_attack_type else None

print("Features Shape :", X.shape)
print("Labels Shape   :", y.shape)
if has_attack_type:
    print("Attack-Type Labels Shape :", y_attack.shape)
else:
    print("No 'attack_type' column found in combined_dataset.csv — "
          "skipping attack-type export (binary-only pipeline).")

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

print("\nSplitting Dataset...")

if has_attack_type:
    X_train, X_test, y_train, y_test, y_attack_train, y_attack_test = train_test_split(
        X,
        y,
        y_attack,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    y_attack_train, y_attack_test = None, None

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

print("\nScaling Features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling Complete")

# --------------------------------------------------
# Save Processed Data
# --------------------------------------------------

import os

os.makedirs("processed", exist_ok=True)

pd.DataFrame(X_train).to_csv("processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("processed/X_test.csv", index=False)

pd.DataFrame(y_train).to_csv("processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("processed/y_test.csv", index=False)

if has_attack_type:
    # reset_index so these line up row-for-row with y_train/y_test
    # (train_test_split preserves alignment across all arrays passed
    # to it together, so this is purely for a clean CSV output).
    pd.DataFrame(y_attack_train.reset_index(drop=True)).to_csv(
        "processed/y_train_attack.csv", index=False
    )
    pd.DataFrame(y_attack_test.reset_index(drop=True)).to_csv(
        "processed/y_test_attack.csv", index=False
    )

print("\nProcessed files saved successfully!")

print("""
Saved Files:
processed/X_train.csv
processed/X_test.csv
processed/y_train.csv
processed/y_test.csv""" + ("""
processed/y_train_attack.csv
processed/y_test_attack.csv""" if has_attack_type else ""))
# --------------------------------------------------
# Save Scaler
# --------------------------------------------------

joblib.dump(scaler, "processed/scaler.pkl")

print("Scaler Saved Successfully!")