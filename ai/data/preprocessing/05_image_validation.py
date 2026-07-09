import os
import json
import pandas as pd

from PIL import Image
from tqdm import tqdm

from ai.configs.config import IMAGE_FOLDERS

# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = "ai/data/processed/clean_metadata.csv"

REPORT_DIR = "ai/data/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Load Metadata
# ==========================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("IMAGE VALIDATION")
print("=" * 60)

# ==========================================================
# Build Image Lookup Table
# ==========================================================

print("Building image index...")

image_lookup = {}

for folder in IMAGE_FOLDERS:

    if not folder.exists():
        continue

    for file in folder.iterdir():

        if file.suffix.lower() == ".png":
            image_lookup[file.name] = file

print(f"Indexed Images : {len(image_lookup)}")

# ==========================================================
# Validation
# ==========================================================

missing_images = []
corrupted_images = []

total_images = len(df)
valid_images = 0

image_widths = []
image_heights = []

print("\nValidating Images...\n")

for image_name in tqdm(df["image_index"]):

    image_path = image_lookup.get(image_name)

    if image_path is None:
        missing_images.append(image_name)
        continue

    try:

        with Image.open(image_path) as img:

            img.verify()

        with Image.open(image_path) as img:

            width, height = img.size

            image_widths.append(width)
            image_heights.append(height)

        valid_images += 1

    except Exception:

        corrupted_images.append(image_name)

# ==========================================================
# Summary
# ==========================================================

summary = {
    "total_images": total_images,
    "valid_images": valid_images,
    "missing_images": len(missing_images),
    "corrupted_images": len(corrupted_images),
    "min_width": min(image_widths) if image_widths else None,
    "max_width": max(image_widths) if image_widths else None,
    "min_height": min(image_heights) if image_heights else None,
    "max_height": max(image_heights) if image_heights else None
}

# ==========================================================
# Save Reports
# ==========================================================

with open(
    os.path.join(REPORT_DIR, "image_validation_report.json"),
    "w"
) as f:

    json.dump(summary, f, indent=4)

pd.DataFrame(
    {"missing_images": missing_images}
).to_csv(
    os.path.join(REPORT_DIR, "missing_images.csv"),
    index=False
)

pd.DataFrame(
    {"corrupted_images": corrupted_images}
).to_csv(
    os.path.join(REPORT_DIR, "corrupted_images.csv"),
    index=False
)

# ==========================================================
# Print Summary
# ==========================================================

print("\n")
print("=" * 60)

print(f"Total Images      : {total_images}")
print(f"Valid Images      : {valid_images}")
print(f"Missing Images    : {len(missing_images)}")
print(f"Corrupted Images  : {len(corrupted_images)}")

if image_widths:

    print(f"Width Range       : {min(image_widths)} - {max(image_widths)}")

if image_heights:

    print(f"Height Range      : {min(image_heights)} - {max(image_heights)}")

print("=" * 60)

print("\nValidation Completed Successfully.")