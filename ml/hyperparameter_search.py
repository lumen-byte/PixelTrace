"""
PixelTrace - Hyperparameter Search
----------------------------------
Tunes XGBoost using RandomizedSearchCV.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from scipy.stats import randint, uniform

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
)

from xgboost import XGBClassifier


class HyperparameterSearch:

    def __init__(self):

        train_selected = Path(
            "outputs/features/features_train_selected.csv"
        )

        if train_selected.exists():

            print("Using selected features.")

            self.df = pd.read_csv(train_selected)

        else:

            print("Using full feature set.")

            self.df = pd.read_csv(
                "outputs/features/features_train.csv"
            )

        self.X = self.df.drop(columns=["label"])
        self.y = self.df["label"]

    def search(self):

        model = XGBClassifier(

            objective="binary:logistic",

            eval_metric="logloss",

            random_state=42,

            n_jobs=-1

        )

        param_grid = {

            "n_estimators": randint(200, 800),

            "max_depth": randint(3, 10),

            "learning_rate": uniform(0.01, 0.2),

            "subsample": uniform(0.6, 0.4),

            "colsample_bytree": uniform(0.6, 0.4),

            "min_child_weight": randint(1, 8),

            "gamma": uniform(0, 5)

        }

        cv = StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=42

        )

        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=param_grid,

            n_iter=50,

            scoring="accuracy",

            cv=cv,

            random_state=42,

            n_jobs=-1,

            verbose=2

        )

        search.fit(self.X, self.y)

        print("\n")

        print("=" * 60)

        print("BEST SCORE")

        print(search.best_score_)

        print("\n")

        print("BEST PARAMETERS")

        print(search.best_params_)

        print("=" * 60)

        Path("ml/models").mkdir(

            parents=True,

            exist_ok=True

        )

        joblib.dump(

            search.best_estimator_,

            "ml/models/xgboost_tuned.pkl"

        )

        Path("outputs").mkdir(

            exist_ok=True

        )

        with open(

            "outputs/best_xgboost_params.json",

            "w"

        ) as f:

            json.dump(

                search.best_params_,

                f,

                indent=4

            )


if __name__ == "__main__":

    HyperparameterSearch().search()
