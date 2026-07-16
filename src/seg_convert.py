"""Convert Supervisely-format polygon annotations into YOLO segmentation format."""

import json
import os
import random
import shutil

RAW_DIR = "data/segmentation/Teeth Segmentation JSON/d2"
YOLO_SEG_DIR = "yolo_seg_data"
VAL_FRACTION = 0.15
SEED = 42


def collect_classes():
    """Scan all annotation files to build a sorted, consistent tooth-number class list."""
    classes = set()
    ann_dir = f"{RAW_DIR}/ann"
    for fname in os.listdir(ann_dir):
        data = json.load(open(f"{ann_dir}/{fname}"))
        for obj in data["objects"]:
            classes.add(obj["classTitle"])
    return sorted(classes, key=lambda x: int(x) if x.isdigit() else x)


def convert():
    classes = collect_classes()
    class_to_id = {c: i for i, c in enumerate(classes)}

    img_dir = f"{RAW_DIR}/img"
    ann_dir = f"{RAW_DIR}/ann"
    filenames = sorted(os.listdir(img_dir))

    random.Random(SEED).shuffle(filenames)
    val_count = int(len(filenames) * VAL_FRACTION)
    val_files = set(filenames[:val_count])

    for fname in filenames:
        split = "valid" if fname in val_files else "train"
        images_out = f"{YOLO_SEG_DIR}/images/{split}"
        labels_out = f"{YOLO_SEG_DIR}/labels/{split}"
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        shutil.copy(f"{img_dir}/{fname}", f"{images_out}/{fname}")

        data = json.load(open(f"{ann_dir}/{fname}.json"))
        width = data["size"]["width"]
        height = data["size"]["height"]

        lines = []
        for obj in data["objects"]:
            if obj["geometryType"] != "polygon":
                continue
            class_id = class_to_id[obj["classTitle"]]
            points = obj["points"]["exterior"]
            normalized = []
            for x, y in points:
                normalized.append(f"{x / width:.6f}")
                normalized.append(f"{y / height:.6f}")
            lines.append(f"{class_id} " + " ".join(normalized))

        label_path = f"{labels_out}/{os.path.splitext(fname)[0]}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    yaml_content = f"path: {os.path.abspath(YOLO_SEG_DIR)}\ntrain: images/train\nval: images/valid\n\nnames:\n"
    for i, c in enumerate(classes):
        yaml_content += f"  {i}: '{c}'\n"
    with open(f"{YOLO_SEG_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Converted {len(filenames)} images ({len(filenames) - val_count} train, {val_count} valid)")
    print(f"Classes ({len(classes)}):", classes)


if __name__ == "__main__":
    convert()