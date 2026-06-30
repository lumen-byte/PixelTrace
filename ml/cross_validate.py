"""
PixelTrace - Stratified 5-Fold Cross Validation
-----------------------------------------------
Evaluates core classifiers using Stratified 5-Fold CV on training data.
Features are scaled dynamically within each fold to prevent data leakage.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


def main():
    # Load dataset
    train_selected = Path("outputs/features/features_train_selected.csv")
    if train_selected.exists():
        print("Using selected features.")
        train_df = pd.read_csv(train_selected)
    else:
        print("Using full feature set.")
        train_df = pd.read_csv("outputs/features/features_train.csv")

    X = train_df.drop(columns=["label"])
    y = train_df["label"]

    # Define models with exactly the same hyperparameters as train.py
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        ),
        "SVM": SVC(
            C=10.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "MLP_Tuned": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="tanh",
            solver="lbfgs",
            alpha=0.001,
            max_iter=2000,
            random_state=42
        ),
        "Stacking": StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)),
                ("xgb", XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")),
                ("svm", SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=42))
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        )
    }

    # Stratified 5-Fold Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    print("\n" + "=" * 80)
    print("                      STRATIFIED 5-FOLD CROSS VALIDATION")
    print("=" * 80)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        fold_scores = []
        
        # Manual loop to ensure no data leakage in scaling and to track fold scores
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit scaler ONLY on fold training data to prevent leakage
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Fit model on scaled fold training data
            model.fit(X_train_scaled, y_train_fold)
            
            # Evaluate on scaled fold validation data
            preds = model.predict(X_val_scaled)
            acc = accuracy_score(y_val_fold, preds)
            fold_scores.append(acc)
            
            print(f"  Fold {fold + 1}: {acc:.4f}")
            
        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)
        
        print(f"  Mean Accuracy: {mean_acc:.4f} | Std: {std_acc:.4f}")
        
        results.append({
            "Model": name,
            "Mean Accuracy": round(float(mean_acc), 4),
            "Std": round(float(std_acc), 4)
        })

    print("\n" + "=" * 80)

    # Create comparison dataframe and sort by Mean Accuracy descending
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(by="Mean Accuracy", ascending=False)

    # Save to outputs/cross_validation_results.csv
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    comparison.to_csv(output_dir / "cross_validation_results.csv", index=False)

    # Print results
    print(comparison.to_string(index=False))
    print("=" * 80)

    best_model = comparison.iloc[0]
    print(f"\nBest Cross-Validation Model: {best_model['Model']} (Mean Accuracy: {best_model['Mean Accuracy']:.4f} ± {best_model['Std']:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
