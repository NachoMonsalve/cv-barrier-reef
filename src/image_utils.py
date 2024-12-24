# helpers.py
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from collections import Counter
import json
import matplotlib.pyplot as plt
import pandas as pd


def count_images(folder: str) -> int:
    count = 0
    folder_path = Path(folder).resolve()
    for file in folder_path.rglob("*"):
        if file.suffix.lower() in [".jpg", ".png", ".jpeg"]: 
            count += 1
    return count

def analyze_image_dimensions(folder: str, limit: int = 100) -> Dict[Tuple[int, int], int]:
    dimensions = Counter()
    processed = 0
    folder_path = Path(folder).resolve()
    for file in folder_path.rglob("*"):  # Recursively iterate over all files
        if processed >= limit:
            break
        if file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            with Image.open(file) as img:
                dimensions[img.size] += 1
            processed += 1
    return dict(dimensions)

def calculate_space(folder: str) -> Tuple[float, float, int]:
    total_size = 0
    image_count = 0
    folder_path = Path(folder).resolve()
    for file in folder_path.rglob("*"):  # Recursively iterate over all files
        if file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            image_count += 1
            total_size += file.stat().st_size  # Use pathlib's stat() for file size
    total_size_mb = total_size / (1024 * 1024)
    average_size_kb = (total_size / image_count / 1024) if image_count > 0 else 0
    return total_size_mb, average_size_kb, image_count

def filter_rows_with_annotations(df: pd.DataFrame) -> pd.DataFrame: 
    return df[df["annotations"] != "[]"]

def get_image_path(image_folder: str, video_id: int, video_frame: int) -> Path:
    return Path(image_folder).resolve() / f"video_{video_id}" / f"{video_frame}.jpg"

def load_image(image_path: Path):
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None, None, None
    img = Image.open(image_path)
    return img, img.size[0], img.size[1]