"""
PixelTrace - CNN Embedding Builder
----------------------------------
Loads the CNN feature extractor, extracts embeddings from the train and test splits,
applies TruncatedSVD (32 components) fit only on train, and saves the reduced features.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from ml.cnn_embedding import CNNEmbeddingExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def process_split(dataset_dir: Path, extractor: CNNEmbeddingExtractor) -> tuple:
    """
    Extracts raw embeddings and labels from a split directory.

    Args:
        dataset_dir: Path to train or test split directory containing screen/natural subdirectories.
        extractor: CNN feature extractor instance.

    Returns:
        A tuple of (numpy array of embeddings, numpy array of labels).
    """
    classes = {
        "natural": 0,
        "screen": 1
    }

    embeddings = []
    labels = []

    for class_name, label in classes.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Directory not found: {class_dir}")
            continue

        # Get sorted list of images to ensure determinism
        image_paths = sorted([
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if not image_paths:
            logger.warning(f"No images found in {class_dir}")
            continue

        logger.info(f"Extracting CNN embeddings for {class_name} ({len(image_paths)} images) in {dataset_dir.name}...")
        for img_path in tqdm(image_paths, desc=f"{dataset_dir.name}/{class_name}"):
            try:
                emb = extractor.extract(img_path)
                embeddings.append(emb)
                labels.append(label)
            except Exception as e:
                logger.error(f"Failed to extract embedding for {img_path.name}: {e}")

    return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    train_dir = Path("dataset/train")
    test_dir = Path("dataset/test")

    # 1. Initialize CNN extractor
    extractor = CNNEmbeddingExtractor()

    # 2. Extract training embeddings
    logger.info("Processing training split...")
    train_embeddings, train_labels = process_split(train_dir, extractor)
    if len(train_embeddings) == 0:
        logger.error("No training embeddings extracted. Aborting.")
        return

    # 3. Extract testing embeddings
    logger.info("Processing testing split...")
    test_embeddings, test_labels = process_split(test_dir, extractor)
    if len(test_embeddings) == 0:
        logger.error("No testing embeddings extracted. Aborting.")
        return

    logger.info(f"Train embeddings shape: {train_embeddings.shape}")
    logger.info(f"Test embeddings shape: {test_embeddings.shape}")

    # 4. Fit TruncatedSVD on training set ONLY
    n_components = 128
    logger.info(f"Fitting TruncatedSVD with {n_components} components on training embeddings...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    train_reduced = svd.fit_transform(train_embeddings)
    test_reduced = svd.transform(test_embeddings)

    logger.info(f"Explained variance ratio sum: {np.sum(svd.explained_variance_ratio_):.4f}")

    # 5. Format DataFrames
    cols = [f"cnn_{i:03d}" for i in range(n_components)]
    
    train_df = pd.DataFrame(train_reduced, columns=cols)
    train_df["label"] = train_labels

    test_df = pd.DataFrame(test_reduced, columns=cols)
    test_df["label"] = test_labels

    # 6. Save outputs
    features_dir = Path("outputs/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("ml/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = features_dir / "cnn_train.csv"
    test_csv_path = features_dir / "cnn_test.csv"
    svd_model_path = models_dir / "cnn_svd.pkl"

    logger.info(f"Saving reduced features to {train_csv_path} and {test_csv_path}...")
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    logger.info(f"Saving TruncatedSVD model to {svd_model_path}...")
    joblib.dump(svd, svd_model_path)

    logger.info("CNN Embedding build pipeline completed successfully!")


if __name__ == "__main__":
    main()
