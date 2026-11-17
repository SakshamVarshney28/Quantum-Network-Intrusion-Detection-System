import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns

from qnn_model import HybridQNN

print("=" * 60)
print("TESTING HYBRID QNN")
print("=" * 60)

# ---------------------------------------
# Load Test Data
# ---------------------------------------

print("\nLoading Test Data...")

X_test = pd.read_csv("processed/X_test.csv")
y_test = pd.read_csv("processed/y_test.csv")

# Use same subset size as training
X_test = X_test.iloc[:10000]
y_test = y_test.iloc[:10000]

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values.flatten(), dtype=torch.long)

print("Test Samples :", len(X_test))

# ---------------------------------------
# Load Model
# ---------------------------------------

print("\nLoading Saved Model...")

model = HybridQNN()
model.load_state_dict(torch.load("models/hybrid_qnn.pth"))

model.eval()

print("Model Loaded Successfully!")

# ---------------------------------------
# Prediction
# ---------------------------------------

with torch.no_grad():

    outputs = model(X_test)

    _, predictions = torch.max(outputs, 1)

predictions = predictions.numpy()
actual = y_test.numpy()

# ---------------------------------------
# Metrics
# ---------------------------------------

accuracy = accuracy_score(actual, predictions)
precision = precision_score(actual, predictions)
recall = recall_score(actual, predictions)
f1 = f1_score(actual, predictions)

print("\n========== RESULTS ==========")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(actual, predictions))

# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(actual, predictions)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("results/confusion_matrix.png")

print("\nConfusion Matrix Saved!")
print("results/confusion_matrix.png")