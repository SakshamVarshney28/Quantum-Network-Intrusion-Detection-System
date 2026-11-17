import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from qnn_model import HybridQNN

print("=" * 60)
print("TRAINING HYBRID QNN")
print("=" * 60)

# --------------------------------------------------
# Load Processed Dataset
# --------------------------------------------------

print("\nLoading Training Data...")

X_train = pd.read_csv("processed/X_train.csv")
y_train = pd.read_csv("processed/y_train.csv")

X_test = pd.read_csv("processed/X_test.csv")
y_test = pd.read_csv("processed/y_test.csv")

print("Training :", X_train.shape)
print("Testing  :", X_test.shape)

# --------------------------------------------------
# Reduce Dataset Size
# --------------------------------------------------

TRAIN_SAMPLES = 150000
TEST_SAMPLES = 30000

X_train = X_train.iloc[:TRAIN_SAMPLES]
y_train = y_train.iloc[:TRAIN_SAMPLES]

X_test = X_test.iloc[:TEST_SAMPLES]
y_test = y_test.iloc[:TEST_SAMPLES]

print("\nUsing Reduced Dataset")
print("Training :", X_train.shape)
print("Testing  :", X_test.shape)

# --------------------------------------------------
# Convert to Tensor
# --------------------------------------------------

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)

y_train = torch.tensor(
    y_train.values.flatten(),
    dtype=torch.long
)

y_test = torch.tensor(
    y_test.values.flatten(),
    dtype=torch.long
)

print("\nTensor Conversion Complete")
# --------------------------------------------------
# Create DataLoader
# --------------------------------------------------

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("DataLoaders Ready")

# --------------------------------------------------
# Load Hybrid QNN
# --------------------------------------------------

print("\nLoading Hybrid QNN...")

model = HybridQNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

print("Hybrid QNN Loaded")
print("Loss Function and Optimizer Ready")

# --------------------------------------------------
# Training
# --------------------------------------------------

EPOCHS = 30

train_loss = []

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_loss.append(epoch_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss : {epoch_loss:.4f}")
    # --------------------------------------------------
# Testing
# --------------------------------------------------

print("\nTesting Model...")

model.eval()

predictions = []
actual = []

with torch.no_grad():

    for inputs, labels in test_loader:

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.numpy())

        actual.extend(labels.numpy())

accuracy = accuracy_score(actual, predictions)

print("\n==========================================")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print("==========================================")

# --------------------------------------------------
# Save Model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/hybrid_qnn.pth"
)

print("\nModel Saved Successfully!")
print("Location : models/hybrid_qnn.pth")

# --------------------------------------------------
# Save Loss Graph
# --------------------------------------------------

import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(train_loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig("results/training_loss.png")

print("Training Graph Saved")
print("Location : results/training_loss.png")

print("\n==========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==========================================")