import os
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ai.configs.config import (
    IMAGE_FOLDERS,
    TRAIN_LIST,
    TEST_LIST
)

# ==========================================================
# Paths
# ==========================================================

ENCODED_METADATA = "ai/data/processed/encoded_metadata.csv"
ENCODED_LABELS = "ai/data/processed/encoded_labels.csv"

OUTPUT_DIR = "ai/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load Files
# ==========================================================

print("=" * 60)
print("BUILD TRAINING DATASET")
print("=" * 60)

metadata_df = pd.read_csv(ENCODED_METADATA)
labels_df = pd.read_csv(ENCODED_LABELS)

# ==========================================================
# Merge Metadata + Labels
# ==========================================================

dataset = metadata_df.merge(
    labels_df,
    on="image_index",
    how="inner"
)

print(f"Metadata + Labels : {len(dataset)} samples")

# ==========================================================
# Build Image Lookup
# ==========================================================

print("\nBuilding Image Lookup...")

image_lookup = {}

for folder in IMAGE_FOLDERS:

    if not folder.exists():
        continue

    for file in folder.rglob("*.png"):
        image_lookup[file.name] = str(file)

print(f"Indexed Images : {len(image_lookup)}")

# ==========================================================
# Attach Image Paths
# ==========================================================

print("\nAttaching Image Paths...")

image_paths = []

missing = 0

for image_name in tqdm(dataset["image_index"]):

    path = image_lookup.get(image_name)

    if path is None:
        image_paths.append(None)
        missing += 1
    else:
        image_paths.append(path)

dataset.insert(1, "image_path", image_paths)

# ==========================================================
# Remove Missing Images
# ==========================================================

dataset = dataset.dropna(subset=["image_path"])

print(f"Missing Images Removed : {missing}")

# ==========================================================
# Load Official NIH Split Files
# ==========================================================

print("\nLoading Official NIH Split Lists...")

train_list_df = pd.read_csv(
    TRAIN_LIST,
    header=None,
    names=["image_index"]
)

test_list_df = pd.read_csv(
    TEST_LIST,
    header=None,
    names=["image_index"]
)

print(f"Train+Validation Images : {len(train_list_df)}")
print(f"Test Images             : {len(test_list_df)}")

# ==========================================================
# Build Official Train/Test Sets
# ==========================================================

train_val_dataset = dataset.merge(
    train_list_df,
    on="image_index",
    how="inner"
)

test_dataset = dataset.merge(
    test_list_df,
    on="image_index",
    how="inner"
)

# ==========================================================
# Train / Validation Split
# ==========================================================

train_dataset, validation_dataset = train_test_split(
    train_val_dataset,
    test_size=0.10,
    random_state=42,
    shuffle=True
)

# ==========================================================
# Save Datasets
# ==========================================================

train_path = os.path.join(
    OUTPUT_DIR,
    "train_dataset.csv"
)

validation_path = os.path.join(
    OUTPUT_DIR,
    "validation_dataset.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test_dataset.csv"
)

train_dataset.to_csv(
    train_path,
    index=False
)

validation_dataset.to_csv(
    validation_path,
    index=False
)

test_dataset.to_csv(
    test_path,
    index=False
)

# ==========================================================
# Summary
# ==========================================================

print("\n")
print("=" * 60)
print("DATASET SPLIT SUMMARY")
print("=" * 60)

print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(validation_dataset)}")
print(f"Test Samples       : {len(test_dataset)}")

print("\nSaved Files:")

print(train_path)
print(validation_path)
print(test_path)

print("=" * 60)

print("\nDataset Build Completed Successfully.")