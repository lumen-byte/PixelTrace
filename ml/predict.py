"""
PixelTrace - Prediction Module
------------------------------
Extracts features (handcrafted and CNN embeddings) from a query image,
applies scaling, and predicts class membership (Screen vs. Natural).
"""

from pathlib import Path
import joblib
import pandas as pd

from cv.preprocessing import ImagePreprocessor
from cv.feature_fusion import FeatureFusionEngine
from ml.feature_schema import FEATURE_COLUMNS


class Predictor:

    def __init__(self):
        # Load trained best model, scaler, and CNN SVD projector
        model_dir = Path("ml/models")
        
        self.model = joblib.load(
            model_dir / "best_model.pkl"
        )
        
        self.scaler = joblib.load(
            model_dir / "scaler.pkl"
        )

        # Load SVD projector explicitly to verify it exists
        self.svd_model = joblib.load(
            model_dir / "cnn_svd.pkl"
        )

        self.preprocessor = ImagePreprocessor()
        self.fusion = FeatureFusionEngine()

    def predict(self, image_path: str):
        # 1. Preprocess the image
        data = self.preprocessor.preprocess(image_path)

        # 2. Extract handcrafted and CNN features
        features = self.fusion.extract(data)

        # 3. Align features with the scaler's expected input columns
        row = {}
        for feature in self.scaler.feature_names_in_:
            row[feature] = features.get(feature, 0.0)

        df = pd.DataFrame([row])

        # 4. Scale features using the saved scaler
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns
        )

        # 5. Predict using the trained hybrid model
        prediction = int(self.model.predict(df_scaled)[0])

        confidence = float(
            self.model.predict_proba(df_scaled).max()
        )

        return {
            "prediction": "Screen" if prediction else "Natural",
            "confidence": round(confidence * 100, 2)
        }


if __name__ == "__main__":
    predictor = Predictor()
    image = "dataset/test/screen/screen_0001.jpg"
    print(predictor.predict(image))