import os
import pandas as pd

from tqdm import tqdm

from ai.configs.config import IMAGE_FOLDERS

# ==========================================================
# Paths
# ==========================================================

CLEAN_METADATA = "ai/data/processed/clean_metadata.csv"
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

clean_df = pd.read_csv(CLEAN_METADATA)

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
# Attach Image Path
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
# Save Final Dataset
# ==========================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "training_dataset.csv"
)

dataset.to_csv(
    output_file,
    index=False
)

print(f"\nSaved : {output_file}")

# ==========================================================
# Dataset Summary
# ==========================================================

print("\n")
print("=" * 60)

print(f"Final Samples : {len(dataset)}")

print(f"Total Columns : {len(dataset.columns)}")

print("\nColumns:")

for col in dataset.columns:
    print(col)

print("=" * 60)

print("\nTraining Dataset Built Successfully.")