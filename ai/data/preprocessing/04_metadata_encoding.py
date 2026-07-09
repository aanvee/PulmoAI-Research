import os
import joblib
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = "ai/data/processed/clean_metadata.csv"

OUTPUT_DIR = "ai/data/processed"
REPORT_DIR = "ai/data/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Load Metadata
# ==========================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("METADATA ENCODING")
print("=" * 60)

# ==========================================================
# Select Required Columns
# ==========================================================

metadata_df = pd.DataFrame()

metadata_df["image_index"] = df["image_index"]

# ==========================================================
# Encode Gender
# ==========================================================

gender_map = {
    "M": 1,
    "F": 0
}

metadata_df["patient_gender"] = df["patient_gender"].map(gender_map)

print("Gender Encoding Completed.")

# ==========================================================
# Encode View Position
# ==========================================================

view_map = {
    "AP": 1,
    "PA": 0
}

metadata_df["view_position"] = df["view_position"].map(view_map)

print("View Position Encoding Completed.")

# ==========================================================
# Normalize Age
# ==========================================================

scaler = MinMaxScaler()

metadata_df["patient_age"] = scaler.fit_transform(
    df[["patient_age"]]
)

print("Age Normalization Completed.")

# ==========================================================
# Save Scaler
# ==========================================================

joblib.dump(
    scaler,
    os.path.join(REPORT_DIR, "age_scaler.pkl")
)

print("Scaler Saved.")

# ==========================================================
# Save Encoded Metadata
# ==========================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "encoded_metadata.csv"
)

metadata_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved : {output_file}")

# ==========================================================
# Display Summary
# ==========================================================

print("\nEncoded Metadata Preview")

print(metadata_df.head())

print("\nMetadata Encoding Completed Successfully.")