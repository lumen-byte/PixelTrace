"""
PixelTrace - CNN Feature Extractor
----------------------------------
Extracts embeddings from images using a frozen, pretrained MobileNetV3 Small model.
Supports both CPU/GPU (including Metal Performance Shaders for macOS) automatically.
"""

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms

# Configure logging
import os
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MobileNetV3FeatureExtractor:
    """
    Production-grade CNN Feature Extractor using pretrained MobileNetV3 Small.
    """

    def __init__(self):
        # 1. Automatic device selection (CPU / CUDA / MPS)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using GPU (CUDA) for feature extraction.")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS) for feature extraction.")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for feature extraction.")

        # 2. Initialize pretrained MobileNetV3 Small model without classification head
        try:
            logger.info("Loading pretrained MobileNetV3 Small model...")
            self.model = timm.create_model(
                "mobilenetv3_small_100",
                pretrained=False,
                num_classes=0
            )
            
            # Direct local cache loading to prevent network round-trips
            cache_file = Path.home() / ".cache/huggingface/hub/models--timm--mobilenetv3_small_100.lamb_in1k/snapshots/1824797e7887cbec1990e4adbd6675960a36c589/model.safetensors"
            if cache_file.exists():
                from safetensors.torch import load_file
                state_dict = load_file(cache_file)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded weights from local cache: {cache_file}")
            else:
                logger.warning("Local cache weights not found, attempting online load...")
                self.model = timm.create_model(
                    "mobilenetv3_small_100",
                    pretrained=True,
                    num_classes=0
                )

            self.model = self.model.to(self.device)
            self.model.eval()

            # Freeze all model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            logger.info("Model loaded and parameters frozen successfully.")
        except Exception as e:
            logger.error(f"Failed to load MobileNetV3 model: {e}")
            raise

        # 3. Preprocessing transformation (Resize to 224x224 and ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Extract the feature embedding from an image.

        Args:
            image_input: Absolute file path, Path object, or numpy array (BGR or Grayscale).

        Returns:
            A 1D float32 numpy array representing the embedding vector.

        Raises:
            ValueError: If input format is invalid or cannot be processed.
            FileNotFoundError: If image file path does not exist.
        """
        try:
            pil_image = None

            # Handle file path input
            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {path}")
                
                # Open with PIL and convert to RGB
                pil_image = Image.open(path).convert("RGB")

            # Handle numpy array input
            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    raise ValueError("Received an empty numpy array.")

                # Determine channel configuration and convert to RGB
                if len(image_input.shape) == 2:
                    # Grayscale (H, W)
                    rgb_array = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                elif len(image_input.shape) == 3:
                    h, w, c = image_input.shape
                    if c == 1:
                        rgb_array = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                    elif c == 3:
                        # OpenCV standard BGR to RGB
                        rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                    elif c == 4:
                        rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
                    else:
                        raise ValueError(f"Unsupported number of image channels: {c}")
                else:
                    raise ValueError(f"Invalid numpy array shape: {image_input.shape}")

                pil_image = Image.fromarray(rgb_array)

            else:
                raise ValueError(
                    f"Unsupported input type: {type(image_input)}. "
                    f"Must be a file path string, Path object, or numpy array."
                )

            # Apply standard PyTorch Vision transforms
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Inference (Feature Extraction)
            with torch.no_grad():
                embedding = self.model(tensor)

            # Convert PyTorch tensor to 1D float32 numpy array on CPU
            embedding_np = embedding.cpu().numpy().squeeze(0).astype(np.float32)
            return embedding_np
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            raise


CNNEmbeddingExtractor = MobileNetV3FeatureExtractor


if __name__ == "__main__":
    # Self-test block
    print("Running MobileNetV3FeatureExtractor self-test...")
    extractor = MobileNetV3FeatureExtractor()
    
    # Test with dummy numpy array
    dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    vector = extractor.extract(dummy_img)
    print("Extraction successful!")
    print(f"Embedding shape: {vector.shape}")
    print(f"Embedding dtype: {vector.dtype}")
    print(f"First 5 elements: {vector[:5]}")
