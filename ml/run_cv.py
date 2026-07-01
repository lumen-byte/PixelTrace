"""
PixelTrace - Cross Validation Runner
------------------------------------
Performs Stratified 5-Fold Cross Validation on the training dataset using the 
active handcrafted features (no FFT, no Color, no CNN).
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# 1. Load the generated features
features_path = Path("outputs/features/features_train.csv")
if not features_path.exists():
    print("[ERROR] features_train.csv not found. Please run: PYTHONPATH=. .venv/bin/python ml/build_features.py")
    exit(1)

df = pd.read_csv(features_path)

# Filter out any lingering CNN columns if they exist
feature_cols = [c for c in df.columns if not c.startswith("cnn_") and c != "label" and not c.startswith("fft_") and not c.startswith("moire_") and not c.startswith("h_") and not c.startswith("s_") and not c.startswith("v_") and not c.startswith("corr_") and not c.startswith("lap_") and not c.startswith("gray_") and not c.startswith("sat_")]

print(f"Loaded {len(df)} samples with {len(feature_cols)} handcrafted features.")

X = df[feature_cols].values
y = df["label"].values

# 2. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Define and run the calibrated SVM
model = SVC(C=5.0, kernel="rbf", probability=True, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

print("\n" + "=" * 45)
print("          5-FOLD CROSS VALIDATION RESULTS")
print("=" * 45)
for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score*100:.2f}%")
print("-" * 45)
print(f"Mean Accuracy: {np.mean(scores)*100:.2f}% (std: {np.std(scores)*100:.2f}%)")
print("=" * 45)
