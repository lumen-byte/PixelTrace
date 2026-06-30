"""
PixelTrace - Dataset Preparation Script
----------------------------------------
Converts HEIC raw images to JPG, resizes them, splits them into train/test sets,
and renames them sequentially.
"""

import os
import random
import logging
import shutil
from pathlib import Path
from PIL import Image
import pillow_heif
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

# Define paths
RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")
TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")

CLASSES = ["screen", "natural"]


def clean_and_create_dir(path: Path):
    """Ensure directory exists and is empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def process_class_images(class_name: str) -> list:
    """
    Read HEIC images, convert to JPG, resize to width 512,
    and save them sequentially in the processed folder.
    """
    src_dir = RAW_DIR / class_name
    dest_dir = PROCESSED_DIR / class_name
    
    clean_and_create_dir(dest_dir)
    
    if not src_dir.exists():
        logger.error(f"Source directory {src_dir} does not exist.")
        return []

    # Get all .heic / .HEIC files
    heic_files = sorted([
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".heic", ".heif"]
    ])
    
    if not heic_files:
        logger.warning(f"No HEIC images found in {src_dir}")
        return []

    logger.info(f"Processing {len(heic_files)} raw images for class '{class_name}'...")
    
    processed_paths = []
    
    for i, file_path in enumerate(tqdm(heic_files, desc=f"Converting {class_name}")):
        try:
            # Sequential naming: class_0001.jpg
            new_name = f"{class_name}_{i + 1:04d}.jpg"
            dest_path = dest_dir / new_name
            
            # Open and resize image
            with Image.open(file_path) as img:
                # Maintain aspect ratio
                w, h = img.size
                aspect_ratio = h / w
                new_h = int(512 * aspect_ratio)
                
                resized_img = img.resize((512, new_h), Image.Resampling.LANCZOS)
                
                # Convert to RGB and save as high-quality JPEG
                if resized_img.mode != "RGB":
                    resized_img = resized_img.convert("RGB")
                    
                resized_img.save(dest_path, "JPEG", quality=95)
                
            processed_paths.append(dest_path)
            
        except Exception as e:
            logger.error(f"Failed to process image {file_path.name}: {e}")
            
    return processed_paths


def split_dataset(class_name: str, processed_files: list, split_ratio: float = 0.8):
    """Split processed images into train and test directories."""
    train_dest = TRAIN_DIR / class_name
    test_dest = TEST_DIR / class_name
    
    clean_and_create_dir(train_dest)
    clean_and_create_dir(test_dest)
    
    if not processed_files:
        return 0, 0

    # Shuffle for a random split
    random_files = processed_files.copy()
    random.shuffle(random_files)
    
    split_idx = int(len(random_files) * split_ratio)
    train_set = random_files[:split_idx]
    test_set = random_files[split_idx:]
    
    # Copy files to final destinations
    for file_path in train_set:
        shutil.copy(file_path, train_dest / file_path.name)
        
    for file_path in test_set:
        shutil.copy(file_path, test_dest / file_path.name)
        
    return len(train_set), len(test_set)


def main():
    # Set seed for reproducibility
    random.seed(42)
    
    total_screen = 0
    total_natural = 0
    total_train = 0
    total_test = 0
    
    for class_name in CLASSES:
        # Step 1: Process images (convert, resize, rename sequentially)
        processed_files = process_class_images(class_name)
        
        if class_name == "screen":
            total_screen = len(processed_files)
        else:
            total_natural = len(processed_files)
            
        # Step 2: Split processed images into train (80%) and test (20%)
        train_count, test_count = split_dataset(class_name, processed_files, split_ratio=0.8)
        
        total_train += train_count
        total_test += test_count

    # Print summary
    print("\n" + "=" * 40)
    print("         DATASET PROCESSING SUMMARY")
    print("=" * 40)
    print(f"Total Screen Images:    {total_screen}")
    print(f"Total Natural Images:   {total_natural}")
    print(f"Total Train Images:     {total_train}")
    print(f"Total Test Images:      {total_test}")
    print("=" * 40)


if __name__ == "__main__":
    main()
