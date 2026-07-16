"""Train a YOLO segmentation model on the teeth segmentation dataset."""

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="yolo_seg_data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device="0",
        patience=20,
        project="runs",
        name="dental_seg_gpu",
    )


if __name__ == "__main__":
    main()