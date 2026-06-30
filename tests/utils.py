from pathlib import Path

def get_test_image() -> str:
    """
    Search inside dataset/test/screen/ and return the path of the first .jpg image found.
    Raises FileNotFoundError if no image exists.
    """
    test_dir = Path("dataset/test/screen")
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
    jpg_files = list(test_dir.glob("*.jpg"))
    
    if not jpg_files:
        raise FileNotFoundError(f"No .jpg test images found in: {test_dir}")
        
    return str(jpg_files[0])
