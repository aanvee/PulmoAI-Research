import os
import json
import pandas as pd

from ai.configs.config import METADATA_FILE

# ==========================================================
# Paths
# ==========================================================

PROCESSED_DIR = "ai/data/processed"
REPORT_DIR = "ai/data/reports"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Load Metadata
# ==========================================================

df = pd.read_csv(METADATA_FILE)

print("=" * 60)
print("METADATA CLEANING")
print("=" * 60)

original_rows = len(df)

# ==========================================================
# Remove Unnamed Columns
# ==========================================================

unnamed_cols = [col for col in df.columns if "Unnamed" in col]

if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)

print(f"Removed Unnamed Columns : {len(unnamed_cols)}")

# ==========================================================
# Remove Duplicate Rows
# ==========================================================

duplicate_rows = df.duplicated().sum()

df.drop_duplicates(inplace=True)

print(f"Duplicate Rows Removed : {duplicate_rows}")

# ==========================================================
# Rename Columns
# ==========================================================

df.rename(columns={
    "Image Index": "image_index",
    "Finding Labels": "finding_labels",
    "Follow-up #": "follow_up",
    "Patient ID": "patient_id",
    "Patient Age": "patient_age",
    "Patient Gender": "patient_gender",
    "View Position": "view_position",
    "OriginalImage[Width": "image_width",
    "Height]": "image_height",
    "OriginalImagePixelSpacing[x": "pixel_spacing_x",
    "y]": "pixel_spacing_y"
}, inplace=True)

print("Column names standardized.")

# ==========================================================
# Clean String Columns
# ==========================================================

string_columns = [
    "image_index",
    "finding_labels",
    "patient_gender",
    "view_position"
]

for col in string_columns:
    df[col] = df[col].astype(str).str.strip()

print("String columns cleaned.")

# ==========================================================
# Validate Age
# ==========================================================

invalid_age = df[
    (df["patient_age"] < 0) |
    (df["patient_age"] > 120)
]

print(f"Invalid Age Records : {len(invalid_age)}")

# ==========================================================
# Check Missing Values
# ==========================================================

missing = df.isnull().sum()

print("\nMissing Values")
print(missing)

# ==========================================================
# Save Clean Metadata
# ==========================================================

output_file = os.path.join(
    PROCESSED_DIR,
    "clean_metadata.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved : {output_file}")

# ==========================================================
# Save Cleaning Report
# ==========================================================

report = {
    "original_rows": int(original_rows),
    "final_rows": int(len(df)),
    "duplicate_rows_removed": int(duplicate_rows),
    "unnamed_columns_removed": unnamed_cols,
    "invalid_age_records": int(len(invalid_age)),
    "missing_values": missing.to_dict()
}

with open(
    os.path.join(REPORT_DIR, "cleaning_report.json"),
    "w"
) as f:
    json.dump(report, f, indent=4)

print("\nCleaning report saved.")

print("\nMetadata cleaning completed successfully.")