"""Load dental radiography data and build multi-label targets."""

import pandas as pd

DATA_DIR = "data"
CLASSES = ["Cavity", "Fillings", "Impacted Tooth", "Implant"]


def load_annotations(split: str) -> pd.DataFrame:
    """Load the raw bounding-box annotations CSV for a split (train/valid/test)."""
    return pd.read_csv(f"{DATA_DIR}/{split}/_annotations.csv")


def build_multilabel_targets(annotations: pd.DataFrame) -> pd.DataFrame:
    """Collapse bounding-box annotations into one multi-label row per image.

    Returns a DataFrame with columns: filename, Cavity, Fillings, Impacted Tooth, Implant
    (each 0/1, 1 meaning that class appears somewhere in the image).
    """
    image_classes = annotations.groupby("filename")["class"].apply(set)
    rows = []
    for filename, classes in image_classes.items():
        row = {"filename": filename}
        for cls in CLASSES:
            row[cls] = int(cls in classes)
        rows.append(row)
    return pd.DataFrame(rows)