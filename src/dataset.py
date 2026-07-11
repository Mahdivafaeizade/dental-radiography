"""PyTorch Dataset for loading dental X-ray images with multi-label targets."""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data_loader import CLASSES, DATA_DIR

IMAGE_SIZE = 224

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DentalXrayDataset(Dataset):
    """Loads dental X-ray images and their multi-label targets."""

    def __init__(self, targets_df, split: str, transform=None):
        self.targets_df = targets_df.reset_index(drop=True)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.targets_df)

    def __getitem__(self, idx):
        row = self.targets_df.iloc[idx]
        image_path = f"{DATA_DIR}/{self.split}/{row['filename']}"
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[CLASSES].values.astype("float32"))
        return image, labels