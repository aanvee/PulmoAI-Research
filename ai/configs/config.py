from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================================
# Dataset Root
# Change only this line on different machines
# ==========================================================

DATASET_ROOT = Path(r"D:\Datasets\NIH_ChestXray14")

# ==========================================================
# Original NIH Files
# ==========================================================

METADATA_FILE = DATASET_ROOT / "Data_Entry_2017.csv"

TRAIN_LIST = DATASET_ROOT / "train_val_list.txt"

TEST_LIST = DATASET_ROOT / "test_list.txt"

# ==========================================================
# Processed Dataset Files
# ==========================================================

PROCESSED_DIR = PROJECT_ROOT / "ai" / "data" / "processed"

TRAIN_DATASET = PROCESSED_DIR / "train_dataset.csv"

VALIDATION_DATASET = PROCESSED_DIR / "validation_dataset.csv"

TEST_DATASET = PROCESSED_DIR / "test_dataset.csv"

# ==========================================================
# Image Folders
# ==========================================================

IMAGE_FOLDERS = [
    DATASET_ROOT / f"images_{i:03d}"
    for i in range(1, 13)
]