#!/usr/bin/env python3
"""
Generate a confusion matrix heatmap from detailed classification results.

Requirements:
    pip install pandas scikit-learn matplotlib seaborn
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ======================================================
# Configuration
# ======================================================
CSV_FILE = "./models/test_predictions_inceptionv3.csv"  # path to your CSV
OUTPUT_PNG = "confusion_matrix_inception.png"         # output image file
NORMALIZE = True                                        # set to False for raw counts
FIGSIZE = (14, 12)                                      # adjust for label length
TITLE = "Confusion Matrix — InceptionV3"
FONT_SCALE = 0.9                                        # tweak for readability

# ======================================================
# Load data
# ======================================================
df = pd.read_csv(CSV_FILE)

# Ensure required columns exist
if not {"true_label", "pred_label"}.issubset(df.columns):
    raise ValueError("CSV must include 'true_label' and 'pred_label' columns.")

# Sort labels alphabetically for consistency
labels = sorted(df["true_label"].unique())

# Compute confusion matrix
cm = confusion_matrix(df["true_label"], df["pred_label"], labels=labels)

# Normalize if requested
if NORMALIZE:
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm = cm.round(3)

# ======================================================
# Plot heatmap
# ======================================================
sns.set(style="whitegrid", font_scale=FONT_SCALE)
plt.figure(figsize=FIGSIZE)
ax = sns.heatmap(
    cm,
    annot=True,
    fmt=".2f" if NORMALIZE else "d",
    cmap="viridis",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={'label': 'Proportion' if NORMALIZE else 'Count'}
)

plt.title(TITLE, fontsize=16, weight="bold", pad=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Save and show
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()
print(f"✅ Confusion matrix saved to: {OUTPUT_PNG}")

