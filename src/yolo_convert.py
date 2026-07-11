"""Convert bounding-box CSV annotations into YOLO training format."""

import os
import shutil

import pandas as pd

from src.data_loader import CLASSES, DATA_DIR

YOLO_DIR = "yolo_data"
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}


def convert_split(split: str, yolo_split_name: str):
    """Convert one split (train/valid) into YOLO image+label folders."""
    ann = pd.read_csv(f"{DATA_DIR}/{split}/_annotations.csv")

    images_dir = f"{YOLO_DIR}/images/{yolo_split_name}"
    labels_dir = f"{YOLO_DIR}/labels/{yolo_split_name}"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for filename, group in ann.groupby("filename"):
        src_image = f"{DATA_DIR}/{split}/{filename}"
        dst_image = f"{images_dir}/{filename}"
        shutil.copy(src_image, dst_image)

        label_lines = []
        for _, row in group.iterrows():
            width, height = row["width"], row["height"]
            x_center = (row["xmin"] + row["xmax"]) / 2 / width
            y_center = (row["ymin"] + row["ymax"]) / 2 / height
            box_width = (row["xmax"] - row["xmin"]) / width
            box_height = (row["ymax"] - row["ymin"]) / height
            class_id = CLASS_TO_ID[row["class"]]
            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        label_path = f"{labels_dir}/{os.path.splitext(filename)[0]}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))


def write_data_yaml():
    yaml_content = f"path: {os.path.abspath(YOLO_DIR)}\ntrain: images/train\nval: images/valid\n\nnames:\n"
    for i, cls in enumerate(CLASSES):
        yaml_content += f"  {i}: {cls}\n"

    with open(f"{YOLO_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    convert_split("train", "train")
    convert_split("valid", "valid")
    write_data_yaml()
    print("Conversion complete. YOLO dataset at:", os.path.abspath(YOLO_DIR))