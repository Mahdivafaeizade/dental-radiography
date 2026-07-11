"""Train a multi-label dental X-ray classifier using transfer learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.data_loader import CLASSES, load_annotations, build_multilabel_targets
from src.dataset import DentalXrayDataset, TRAIN_TRANSFORMS, EVAL_TRANSFORMS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model.to(DEVICE)


def compute_pos_weight(targets_df):
    """Weight rare classes higher so the loss doesn't ignore them."""
    counts = targets_df[CLASSES].sum()
    total = len(targets_df)
    pos_weight = (total - counts) / counts
    return torch.tensor(pos_weight.values, dtype=torch.float32).to(DEVICE)


def main():
    train_ann = load_annotations("train")
    train_targets = build_multilabel_targets(train_ann)
    val_ann = load_annotations("valid")
    val_targets = build_multilabel_targets(val_ann)

    train_ds = DentalXrayDataset(train_targets, split="train", transform=TRAIN_TRANSFORMS)
    val_ds = DentalXrayDataset(val_targets, split="valid", transform=EVAL_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = build_model()
    pos_weight = compute_pos_weight(train_targets)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 8
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_ds)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/model.pth")

        marker = " <- best so far, saved" if improved else ""
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}{marker}")

    print(f"\nBest val_loss: {best_val_loss:.4f} — model saved to models/model.pth")

if __name__ == "__main__":
    main()