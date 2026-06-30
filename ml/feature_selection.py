"""
PixelTrace - Feature Selection
------------------------------
Selects the most important handcrafted features
using Random Forest feature importance.
"""

from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class FeatureSelector:

    def __init__(self):

        self.train = pd.read_csv(
            "outputs/features/features_train.csv"
        )

        self.test = pd.read_csv(
            "outputs/features/features_test.csv"
        )

    def run(self, top_k=15):

        X_train = self.train.drop(columns=["label"])
        y_train = self.train["label"]

        X_test = self.test.drop(columns=["label"])
        y_test = self.test["label"]

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        importance = pd.DataFrame({

            "Feature": X_train.columns,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        Path("outputs").mkdir(
            exist_ok=True
        )

        importance.to_csv(
            "outputs/feature_importance.csv",
            index=False
        )

        selected = importance.head(top_k)["Feature"].tolist()

        train_selected = self.train[selected + ["label"]]

        test_selected = self.test[selected + ["label"]]

        train_selected.to_csv(

            "outputs/features/features_train_selected.csv",

            index=False

        )

        test_selected.to_csv(

            "outputs/features/features_test_selected.csv",

            index=False

        )

        print("\n")

        print("=" * 60)

        print("Top Selected Features")

        print("=" * 60)

        print(importance.head(top_k))

        print("=" * 60)


if __name__ == "__main__":

    FeatureSelector().run()