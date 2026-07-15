"""Train a YOLO object detection model on the dental X-ray dataset."""

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="yolo_data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="0",
        patience=20,
        project="runs",
        name="dental_yolo_gpu",
    )


if __name__ == "__main__":
    main()