"""
AI Project - Classification on the Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository (ID: 17)
Models: SVC, Decision Tree Classifier
(TensorFlow Sequential model is provided separately in tensorflow_model.py)

Author: Kamashany
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1) Load dataset
# ---------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
class_names = data.target_names  # ['malignant', 'benign']

print("Dataset: Breast Cancer Wisconsin (Diagnostic) via sklearn.datasets.load_breast_cancer")
print("Samples:", X.shape[0], " Features:", X.shape[1])
print("Classes:", list(class_names))
print(y.value_counts().to_dict())

# ---------------------------------------------------------------
# 2) Train / test split + scaling
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

np.savez("bc_split.npz",
         X_train=X_train_s, X_test=X_test_s,
         y_train=y_train.values, y_test=y_test.values)
X.assign(target=y).to_csv("breast_cancer_dataset.csv", index=False)

# =================================================================
# MODEL 1: SVC (Support Vector Classifier)
# =================================================================
svc = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
svc.fit(X_train_s, y_train)
y_pred_svc = svc.predict(X_test_s)
acc_svc = accuracy_score(y_test, y_pred_svc)

print("\n================ SVC RESULTS ================")
print(f"Test Accuracy: {acc_svc:.4f}")
print(classification_report(y_test, y_pred_svc, target_names=class_names))

# =================================================================
# MODEL 2: Decision Tree Classifier
# =================================================================
dtc = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
dtc.fit(X_train_s, y_train)
y_pred_dtc = dtc.predict(X_test_s)
acc_dtc = accuracy_score(y_test, y_pred_dtc)

print("\n================ DTC RESULTS ================")
print(f"Test Accuracy: {acc_dtc:.4f}")
print(classification_report(y_test, y_pred_dtc, target_names=class_names))

# ---------------------------------------------------------------
# Save results summary
# ---------------------------------------------------------------
with open("results_summary.txt", "w") as f:
    f.write(f"SVC Test Accuracy: {acc_svc:.4f}\n")
    f.write(f"DTC Test Accuracy: {acc_dtc:.4f}\n")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_svc),
                        display_labels=class_names).plot(ax=axes[0], colorbar=False, cmap="Purples")
axes[0].set_title(f"SVC Confusion Matrix (acc={acc_svc:.2f})")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dtc),
                        display_labels=class_names).plot(ax=axes[1], colorbar=False, cmap="Oranges")
axes[1].set_title(f"DTC Confusion Matrix (acc={acc_dtc:.2f})")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.bar(["SVC", "Decision Tree"], [acc_svc, acc_dtc], color=["#6C3FC5", "#D4A017"])
plt.ylim(0, 1.05)
plt.ylabel("Test Accuracy")
plt.title("Model Accuracy Comparison - Breast Cancer Dataset")
for i, v in enumerate([acc_svc, acc_dtc]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.close()

plt.figure(figsize=(16, 8))
plot_tree(dtc, feature_names=X.columns, class_names=class_names,
          filled=True, rounded=True, fontsize=7)
plt.title("Decision Tree Structure (max_depth=4)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4.5))
plt.bar(["malignant", "benign"], [212, 357], color=["#3E1F73", "#D4A017"])
plt.ylabel("Number of Samples")
plt.title("Class Distribution - Breast Cancer Dataset")
for i, v in enumerate([212, 357]):
    plt.text(i, v + 5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()

print("\nSaved: confusion_matrices.png, accuracy_comparison.png, decision_tree.png, class_distribution.png,")
print("       bc_split.npz, breast_cancer_dataset.csv, results_summary.txt")
