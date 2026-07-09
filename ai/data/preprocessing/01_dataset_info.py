import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from ai.configs.config import METADATA_FILE

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(METADATA_FILE)

print("=" * 60)
print("NIH ChestX-ray14 Dataset")
print("=" * 60)

print(f"Total Images      : {len(df)}")
print(f"Total Patients    : {df['Patient ID'].nunique()}")
print(f"Columns           : {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Images:")
print(df["Image Index"].duplicated().sum())

print("\nDuplicate Patients:")
print(df["Patient ID"].duplicated().sum())

# ==========================
# Disease Statistics
# ==========================

disease_count = {}

for labels in df["Finding Labels"]:
    for disease in labels.split("|"):
        disease_count[disease] = disease_count.get(disease, 0) + 1

disease_df = (
    pd.DataFrame(
        disease_count.items(),
        columns=["Disease", "Count"]
    )
    .sort_values("Count", ascending=False)
)

print("\nDisease Distribution")
print(disease_df)

# ==========================
# Create Reports Folder
# ==========================

REPORT_DIR = "ai/data/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================
# Save Disease CSV
# ==========================

disease_df.to_csv(
    os.path.join(REPORT_DIR, "disease_distribution.csv"),
    index=False
)

# ==========================
# Save Summary JSON
# ==========================

summary = {
    "total_images": int(len(df)),
    "total_patients": int(df["Patient ID"].nunique()),
    "columns": df.columns.tolist(),
    "missing_values": df.isnull().sum().to_dict(),
    "duplicate_images": int(df["Image Index"].duplicated().sum()),
    "duplicate_patients": int(df["Patient ID"].duplicated().sum())
}

with open(
    os.path.join(REPORT_DIR, "dataset_summary.json"),
    "w"
) as f:
    json.dump(summary, f, indent=4)

# ==========================
# Age Distribution
# ==========================

plt.figure(figsize=(8,5))

df["Patient Age"].hist(
    bins=30
)

plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "age_distribution.png")
)

plt.close()

# ==========================
# Gender Distribution
# ==========================

plt.figure(figsize=(6,5))

df["Patient Gender"].value_counts().plot(
    kind="bar"
)

plt.title("Gender Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "gender_distribution.png")
)

plt.close()

# ==========================
# View Position
# ==========================

plt.figure(figsize=(6,5))

df["View Position"].value_counts().plot(
    kind="bar"
)

plt.title("View Position Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "view_position_distribution.png")
)

plt.close()

# ==========================
# Disease Distribution
# ==========================

plt.figure(figsize=(12,6))

plt.bar(
    disease_df["Disease"],
    disease_df["Count"]
)

plt.xticks(rotation=90)

plt.title("Disease Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "disease_distribution.png")
)

plt.close()

print("\nAll reports saved successfully.")