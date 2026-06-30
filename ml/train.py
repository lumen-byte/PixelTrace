"""
PixelTrace - Model Training (v2)
---------------------------------
Trains a suite of models on the FULL feature set (handcrafted + CNN embeddings).
Key improvements over v1:
  - Uses the full feature set (no aggressive feature selection pruning)
  - Replaces deprecated SVC(probability=True) with CalibratedClassifierCV
  - Adds a SoftVoting ensemble of the top 3 models
  - Selects best model by cross-validation score (not hold-out), to prevent
    over-selection on a tiny 40-sample test set
  - Uses RobustScaler instead of StandardScaler (less sensitive to outlier features)
"""

from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self):
        # Prefer full feature set — bypass aggressive feature selection
        train_full = Path("outputs/features/features_train.csv")
        test_full = Path("outputs/features/features_test.csv")

        if not train_full.exists():
            raise FileNotFoundError(
                "features_train.csv not found. Run ml.build_features first."
            )

        logger.info("Loading full feature set (bypassing aggressive feature selection).")
        self.train_df = pd.read_csv(train_full)
        self.test_df = pd.read_csv(test_full)

        cnn_cols = [c for c in self.train_df.columns if c.startswith("cnn_")]
        logger.info(f"Feature set: {self.train_df.shape[1]-1} features ({len(cnn_cols)} CNN components)")

        # Build calibrated SVM (replaces deprecated SVC(probability=True))
        _svm_base = SVC(C=100, gamma="scale", kernel="rbf", random_state=42)
        _svm_calibrated = CalibratedClassifierCV(_svm_base, cv=5, ensemble=False)

        _xgb = XGBClassifier(
            n_estimators=800,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )

        _rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        _mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )

        _lr = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )

        # Soft-voting ensemble of top performers
        _ensemble = VotingClassifier(
            estimators=[
                ("svm", _svm_calibrated),
                ("xgb", _xgb),
                ("mlp", _mlp),
            ],
            voting="soft",
            n_jobs=-1,
        )

        self.models = {
            "XGBoost": _xgb,
            "SVM_Calibrated": _svm_calibrated,
            "RandomForest": _rf,
            "MLP": _mlp,
            "LogisticRegression": _lr,
            "SoftVoting_Ensemble": _ensemble,
        }

    def train(self):
        X_train = self.train_df.drop(columns=["label"])
        y_train = self.train_df["label"]

        X_test = self.test_df.drop(columns=["label"])
        y_test = self.test_df["label"]

        model_dir = Path("ml/models")
        model_dir.mkdir(parents=True, exist_ok=True)

        # RobustScaler: less sensitive to outlier features than StandardScaler
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns
        )

        joblib.dump(scaler, model_dir / "scaler.pkl")

        results = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_model = None
        best_name = ""
        best_cv_score = 0.0

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # 5-fold CV on training data for model selection (avoid overfitting to 40-sample test)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1
            )
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            logger.info(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")

            # Full fit on training set
            model.fit(X_train_scaled, y_train)

            # Evaluate on held-out test set
            test_preds = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_preds)
            logger.info(f"  Test: {test_acc:.4f}")

            joblib.dump(model, model_dir / f"{name}.pkl")

            results.append({
                "Model": name,
                "CV_Mean": round(cv_mean, 4),
                "CV_Std": round(cv_std, 4),
                "Test_Accuracy": round(test_acc, 4),
            })

            # Select best by CV mean (most reliable with n=160 samples)
            if cv_mean > best_cv_score:
                best_cv_score = cv_mean
                best_model = model
                best_name = name

        joblib.dump(best_model, model_dir / "best_model.pkl")

        comparison = pd.DataFrame(results).sort_values("CV_Mean", ascending=False)

        Path("outputs").mkdir(exist_ok=True)
        comparison.to_csv("outputs/model_comparison.csv", index=False)

        info = {
            "best_model": best_name,
            "cv_accuracy": round(best_cv_score, 4),
            "training_samples": len(self.train_df),
            "testing_samples": len(self.test_df),
            "feature_count": X_train.shape[1],
        }

        with open("outputs/best_model_info.json", "w") as f:
            json.dump(info, f, indent=4)

        logger.info("=" * 60)
        logger.info(f"Best Model : {best_name}")
        logger.info(f"CV Accuracy: {best_cv_score:.4f}")
        logger.info("=" * 60)

        print("\n")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()