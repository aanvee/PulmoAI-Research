import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from ai.data.preprocessing.transforms import (
    train_transform,
    valid_transform
)

# ==========================================================
# Dataset
# ==========================================================

class PulmoDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

        self.metadata_columns = [
            "patient_gender",
            "view_position",
            "patient_age"
        ]

        self.label_columns = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
            "Emphysema",
            "Fibrosis",
            "Hernia",
            "Infiltration",
            "Mass",
            "No Finding",
            "Nodule",
            "Pleural_Thickening",
            "Pneumonia",
            "Pneumothorax"
        ]

    # ======================================================

    def __len__(self):

        return len(self.data)

    # ======================================================

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        metadata = torch.tensor(
            [float(x) for x in row[self.metadata_columns]],
            dtype=torch.float32
        )

        labels = torch.tensor(
            [float(x) for x in row[self.label_columns]],
            dtype=torch.float32
        )

        return image, metadata, labels


# ==========================================================
# DataLoader Factory
# ==========================================================

def create_dataloaders(
    train_csv,
    valid_csv=None,
    batch_size=32,
    num_workers=4
):

    train_dataset = PulmoDataset(
        train_csv,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    if valid_csv is None:
        return train_loader

    valid_dataset = PulmoDataset(
        valid_csv,
        transform=valid_transform
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader