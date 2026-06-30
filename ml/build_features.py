"""
PixelTrace - Feature Builder
----------------------------
Extracts handcrafted features from all images and
merges them with CNN embeddings to create final train/test CSV files.
"""

from pathlib import Path
import logging

import pandas as pd
from tqdm import tqdm

from cv.preprocessing import ImagePreprocessor
from cv.feature_fusion import FeatureFusionEngine
from ml.feature_schema import FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class FeatureBuilder:

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.fusion = FeatureFusionEngine()

    def process_directory(self, dataset_path: str):
        rows = []
        classes = {
            "screen": 1,
            "natural": 0
        }

        # Select only handcrafted features for the initial DataFrame
        handcrafted_cols = [
            c for c in FEATURE_COLUMNS 
            if not c.startswith("cnn_") and c != "label"
        ]

        for class_name, label in classes.items():
            folder = Path(dataset_path) / class_name
            images = sorted(folder.glob("*.jpg"))

            logger.info(
                f"{class_name}: {len(images)} images"
            )

            for image in tqdm(images):
                try:
                    data = self.preprocessor.preprocess(
                        str(image)
                    )

                    features = self.fusion.extract(data)

                    # Ensure every feature exists
                    row = {}
                    for feature in handcrafted_cols:
                        row[feature] = features.get(
                            feature,
                            0
                        )

                    row["label"] = label
                    rows.append(row)

                except Exception as e:
                    logger.warning(
                        f"Skipping {image.name}: {e}"
                    )

        return pd.DataFrame(rows)

    def build(self):
        logger.info("Building Training Features")
        train_df = self.process_directory(
            "dataset/train"
        )

        logger.info("Building Testing Features")
        test_df = self.process_directory(
            "dataset/test"
        )

        output = Path("outputs/features")
        output.mkdir(
            parents=True,
            exist_ok=True
        )

        # 1. Save temporary/intermediate handcrafted features
        train_df.to_csv(
            output / "features_train.csv",
            index=False
        )

        test_df.to_csv(
            output / "features_test.csv",
            index=False
        )

        # 2. Automatically merge outputs/features/cnn_train.csv and outputs/features/features_train.csv by row order
        cnn_train_path = output / "cnn_train.csv"
        cnn_test_path = output / "cnn_test.csv"

        if cnn_train_path.exists():
            logger.info("Merging cnn_train.csv with features_train.csv by row order...")
            cnn_train = pd.read_csv(cnn_train_path)
            
            # Extract CNN features (exclude label from CNN dataframe)
            cnn_features = cnn_train.drop(columns=["label"], errors="ignore")
            
            # Align row count verification
            if len(train_df) == len(cnn_features):
                label_col = train_df["label"]
                train_hc_only = train_df.drop(columns=["label"], errors="ignore")
                
                # Merge columns — use copy() to defragment before label assignment
                train_final = pd.concat([train_hc_only, cnn_features], axis=1).copy()
                train_final["label"] = label_col.values
                
                # Overwrite final features CSV
                train_final.to_csv(output / "features_train.csv", index=False)
                train_df = train_final
            else:
                logger.error(
                    f"Row count mismatch: train_df ({len(train_df)}) vs cnn_train ({len(cnn_features)}). Merge skipped."
                )

        if cnn_test_path.exists():
            logger.info("Merging cnn_test.csv with features_test.csv by row order...")
            cnn_test = pd.read_csv(cnn_test_path)
            
            # Extract CNN features (exclude label from CNN dataframe)
            cnn_features = cnn_test.drop(columns=["label"], errors="ignore")
            
            # Align row count verification
            if len(test_df) == len(cnn_features):
                label_col = test_df["label"]
                test_hc_only = test_df.drop(columns=["label"], errors="ignore")
                
                # Merge columns
                test_final = pd.concat([test_hc_only, cnn_features], axis=1)
                test_final["label"] = label_col
                
                # Overwrite final features CSV
                test_final.to_csv(output / "features_test.csv", index=False)
                test_df = test_final
            else:
                logger.error(
                    f"Row count mismatch: test_df ({len(test_df)}) vs cnn_test ({len(cnn_features)}). Merge skipped."
                )

        logger.info("Done.")
        logger.info(
            f"Train Shape : {train_df.shape}"
        )
        logger.info(
            f"Test Shape : {test_df.shape}"
        )


if __name__ == "__main__":
    FeatureBuilder().build()