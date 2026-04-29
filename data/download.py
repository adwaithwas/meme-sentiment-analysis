"""
Dataset download and setup utilities.

Handles downloading the Memotion Dataset 7K from Kaggle,
extracting it, and organizing the file structure.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path


def check_kaggle_credentials():
    """
    Check if Kaggle API credentials are properly configured.
    Returns True if kaggle.json exists in the expected location.
    """
    # Windows: C:\Users\<username>\.kaggle\kaggle.json
    # Linux/Mac: ~/.kaggle/kaggle.json
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print(f"✓ Kaggle credentials found at: {kaggle_json}")
        return True
    else:
        print(f"✗ Kaggle credentials NOT found at: {kaggle_json}")
        print("\n" + "=" * 60)
        print("HOW TO SET UP KAGGLE API:")
        print("=" * 60)
        print("1. Go to https://www.kaggle.com")
        print("2. Click your profile icon (top right) → Settings")
        print("3. Scroll to 'API' section → Click 'Create New Token'")
        print("4. This downloads 'kaggle.json'")
        print(f"5. Create folder: {kaggle_dir}")
        print(f"6. Move kaggle.json to: {kaggle_json}")
        print("=" * 60)
        return False


def download_memotion_dataset(data_dir: str, kaggle_dataset: str = "williamscott701/memotion-dataset-7k"):
    """
    Download the Memotion Dataset 7K from Kaggle.
    
    Args:
        data_dir: Root data directory (will create 'raw' subdirectory)
        kaggle_dataset: Kaggle dataset identifier
    """
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Check if dataset already exists
    expected_csv = os.path.join(raw_dir, "labels.csv")
    expected_images = os.path.join(raw_dir, "images")
    
    if os.path.exists(expected_csv) and os.path.exists(expected_images):
        num_images = len([f for f in os.listdir(expected_images) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if num_images > 100:
            print(f"✓ Dataset already exists: {num_images} images found")
            return True
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        print("\n⚠ Cannot download without Kaggle credentials.")
        print("  Please set up Kaggle API and run again.")
        print(f"\n  Or manually download from:")
        print(f"  https://www.kaggle.com/datasets/{kaggle_dataset}")
        print(f"  Extract contents to: {raw_dir}")
        return False
    
    # Download using Kaggle API
    print(f"\n📥 Downloading dataset: {kaggle_dataset}")
    print(f"   Destination: {raw_dir}")
    
    try:
        # On Windows within venv, kaggle.exe is in the Scripts folder
        kaggle_cmd = "kaggle"
        if os.name == 'nt' and hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            kaggle_cmd = os.path.join(os.path.dirname(sys.executable), "kaggle.exe")
            
        subprocess.run(
            [kaggle_cmd, "datasets", "download",
             "-d", kaggle_dataset, "-p", raw_dir, "--unzip"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e.stderr}")
        print(f"\n  Manual download URL:")
        print(f"  https://www.kaggle.com/datasets/{kaggle_dataset}")
        return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        return download_memotion_dataset(data_dir, kaggle_dataset)
    
    # Organize the downloaded files
    _organize_dataset(raw_dir)
    return True


def _organize_dataset(raw_dir: str):
    """
    Organize the downloaded dataset into a consistent structure.
    
    Expected final structure:
        raw/
        ├── labels.csv        (main annotations file)
        └── images/           (all meme images)
            ├── image_1.jpg
            ├── image_2.jpg
            └── ...
    """
    print("\n📂 Organizing dataset structure...")
    
    # The Memotion dataset might have nested directories after extraction
    # Look for CSV files and image directories
    csv_files = []
    image_dirs = []
    
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, f))
        for d in dirs:
            full_path = os.path.join(root, d)
            # Check if directory contains images
            if any(img.lower().endswith(('.jpg', '.jpeg', '.png')) 
                   for img in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, img))):
                image_dirs.append(full_path)
    
    # Move CSV to raw/labels.csv if not already there
    target_csv = os.path.join(raw_dir, "labels.csv")
    if not os.path.exists(target_csv) and csv_files:
        # Use the largest CSV (likely the main annotations)
        main_csv = max(csv_files, key=os.path.getsize)
        shutil.copy2(main_csv, target_csv)
        print(f"  ✓ Labels: {os.path.basename(main_csv)} → labels.csv")
    
    # Ensure images directory exists at raw/images/
    target_images = os.path.join(raw_dir, "images")
    if not os.path.exists(target_images):
        os.makedirs(target_images, exist_ok=True)
    
    # Move images from nested dirs to raw/images/
    for img_dir in image_dirs:
        if os.path.abspath(img_dir) == os.path.abspath(target_images):
            continue
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(img_dir, img_file)
                dst = os.path.join(target_images, img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    
    # Count results
    num_images = len([f for f in os.listdir(target_images) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  ✓ Images organized: {num_images} files")
    
    if os.path.exists(target_csv):
        import pandas as pd
        df = pd.read_csv(target_csv)
        print(f"  ✓ Labels loaded: {len(df)} entries, {len(df.columns)} columns")
        print(f"  ✓ Columns: {list(df.columns)}")


def verify_dataset(data_dir: str) -> bool:
    """
    Verify that the dataset is properly set up and ready for use.
    
    Returns True if everything looks good.
    """
    raw_dir = os.path.join(data_dir, "raw")
    labels_path = os.path.join(raw_dir, "labels.csv")
    images_dir = os.path.join(raw_dir, "images")
    
    issues = []
    
    if not os.path.exists(labels_path):
        issues.append(f"Labels file not found: {labels_path}")
    
    if not os.path.exists(images_dir):
        issues.append(f"Images directory not found: {images_dir}")
    else:
        num_images = len([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if num_images == 0:
            issues.append("No image files found in images directory")
        else:
            print(f"✓ Found {num_images} images")
    
    if issues:
        print("\n⚠ Dataset verification FAILED:")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    
    print("✓ Dataset verification PASSED")
    return True


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import Config
    
    config = Config()
    print(f"Data directory: {config.data_dir}")
    
    success = download_memotion_dataset(config.data_dir, config.kaggle_dataset)
    if success:
        verify_dataset(config.data_dir)
