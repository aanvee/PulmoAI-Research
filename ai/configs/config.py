from pathlib import Path

# Change only this line on different machines
DATASET_ROOT = Path(r"D:\Datasets\NIH_ChestXray14")

METADATA_FILE = DATASET_ROOT / "Data_Entry_2017.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DATASET = PROJECT_ROOT / "ai" / "data" / "processed" / "training_dataset.csv"

TRAIN_LIST = DATASET_ROOT / "train_val_list.txt"
TEST_LIST = DATASET_ROOT / "test_list.txt"

IMAGE_FOLDERS = [
    DATASET_ROOT / f"images_{i:03d}"
    for i in range(1, 13)
]