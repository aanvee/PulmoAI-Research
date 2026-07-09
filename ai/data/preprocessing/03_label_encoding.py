import os
import json
import pandas as pd

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
print("LABEL ENCODING")
print("=" * 60)

# ==========================================================
# Extract Unique Diseases
# ==========================================================

all_diseases = set()

for labels in df["finding_labels"]:
    diseases = labels.split("|")
    all_diseases.update(diseases)

all_diseases = sorted(all_diseases)

print(f"Total Diseases : {len(all_diseases)}")

print("\nDisease List")

for disease in all_diseases:
    print(disease)

# ==========================================================
# Disease Mapping
# ==========================================================

disease_mapping = {
    disease: idx
    for idx, disease in enumerate(all_diseases)
}

# ==========================================================
# Multi-Hot Encoding
# ==========================================================

encoded_df = pd.DataFrame()

encoded_df["image_index"] = df["image_index"]

for disease in all_diseases:
    encoded_df[disease] = 0

for i, labels in enumerate(df["finding_labels"]):

    diseases = labels.split("|")

    for disease in diseases:
        encoded_df.at[i, disease] = 1

print("\nLabel Encoding Completed.")

# ==========================================================
# Save Encoded Labels
# ==========================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "encoded_labels.csv"
)

encoded_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved : {output_file}")

# ==========================================================
# Save Disease Mapping
# ==========================================================

mapping_file = os.path.join(
    REPORT_DIR,
    "disease_mapping.json"
)

with open(mapping_file, "w") as f:
    json.dump(
        disease_mapping,
        f,
        indent=4
    )

print("Disease Mapping Saved.")

# ==========================================================
# Save Label Statistics
# ==========================================================

label_counts = encoded_df.drop(columns=["image_index"]).sum()

label_stats = (
    label_counts
    .sort_values(ascending=False)
    .to_dict()
)

stats_file = os.path.join(
    REPORT_DIR,
    "label_statistics.json"
)

with open(stats_file, "w") as f:
    json.dump(
        label_stats,
        f,
        indent=4
    )

print("Label Statistics Saved.")

print("\nLabel Encoding Completed Successfully.")