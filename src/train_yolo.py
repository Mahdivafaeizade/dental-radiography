"""Train a YOLO object detection model on the dental X-ray dataset."""

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="yolo_data/data.yaml",
        epochs=30,
        imgsz=416,
        batch=8,
        device="cpu",
        project="runs",
        name="dental_yolo",
    )


if __name__ == "__main__":
    main()