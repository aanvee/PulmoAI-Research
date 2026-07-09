from ai.data.preprocessing.dataloader import PulmoDataset
from ai.data.preprocessing.transforms import train_transform

dataset = PulmoDataset(
    "ai/data/processed/training_dataset.csv",
    transform=train_transform
)

image, metadata, labels = dataset[0]

print(image.shape)
print(metadata.shape)
print(labels.shape)

print(metadata)
print(labels)