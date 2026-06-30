"""
PixelTrace - Model Evaluation
-----------------------------
Evaluates all trained models and generates metrics.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class ModelEvaluator:

    def __init__(self):

        test_selected = Path(
            "outputs/features/features_test_selected.csv"
        )

        if test_selected.exists():
            self.test_df = pd.read_csv(test_selected)
        else:
            self.test_df = pd.read_csv(
                "outputs/features/features_test.csv"
            )

        self.models_dir = Path("ml/models")

        self.X_test = self.test_df.drop(columns=["label"])
        self.y_test = self.test_df["label"]

        # Load standard scaler and scale X_test
        scaler_file = self.models_dir / "scaler.pkl"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
            self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)

    def evaluate(self):

        model_files = sorted(
            self.models_dir.glob("*.pkl")
        )

        results = []

        print("\n" + "=" * 70)

        for model_file in model_files:

            if model_file.name in ["scaler.pkl", "cnn_svd.pkl"] or model_file.name.endswith("_hc.pkl"):
                continue

            model = joblib.load(model_file)

            predictions = model.predict(self.X_test)

            accuracy = accuracy_score(
                self.y_test,
                predictions
            )

            precision = precision_score(
                self.y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                self.y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                self.y_test,
                predictions,
                zero_division=0
            )

            print(f"\n{model_file.stem}")
            print("-" * 50)

            print(
                classification_report(
                    self.y_test,
                    predictions,
                    zero_division=0
                )
            )

            print("Confusion Matrix")

            print(
                confusion_matrix(
                    self.y_test,
                    predictions
                )
            )

            results.append({

                "Model": model_file.stem,
                "Accuracy": round(accuracy,4),
                "Precision": round(precision,4),
                "Recall": round(recall,4),
                "F1": round(f1,4)

            })

        comparison = pd.DataFrame(results)

        comparison.to_csv(
            "outputs/evaluation.csv",
            index=False
        )

        print("=" * 70)
        print(comparison)
        print("=" * 70)


if __name__ == "__main__":

    ModelEvaluator().evaluate()