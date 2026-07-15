"""Run inference with the trained YOLO detector and draw bounding boxes."""

from ultralytics import YOLO

MODEL_PATH = "models/yolo_model.pt"

_model = None


def load_detector():
    """Load the trained YOLO model once and reuse it (avoids reloading on every call)."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def detect(image, conf: float = 0.25):
    """Run detection on an image (path, PIL image, or numpy array), return the result."""
    model = load_detector()
    results = model.predict(image, conf=conf, verbose=False)
    return results[0]