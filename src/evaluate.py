"""Evaluate the trained multi-label dental X-ray classifier."""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.data_loader import CLASSES, load_annotations, build_multilabel_targets
from src.dataset import DentalXrayDataset, EVAL_TRANSFORMS
from src.train import build_model, DEVICE


def evaluate(split: str = "valid", threshold: float = 0.5):
    ann = load_annotations(split)
    targets = build_multilabel_targets(ann)
    dataset = DentalXrayDataset(targets, split=split, transform=EVAL_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=16)

    model = build_model()
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))


if __name__ == "__main__":
    evaluate()