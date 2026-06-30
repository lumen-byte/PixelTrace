"""
PixelTrace - Compare Models
"""

import pandas as pd


comparison = pd.read_csv(
    "outputs/model_comparison.csv"
)

comparison = comparison.sort_values(
    "Accuracy",
    ascending=False
)

print("\n")

print("=" * 60)

print(comparison)

print("=" * 60)

print("\nBest Model")

print(comparison.iloc[0])