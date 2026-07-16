"""Run inference with the trained YOLO segmentation model."""

from ultralytics import YOLO

SEG_MODEL_PATH = "models/seg_model.pt"

_seg_model = None


def load_segmenter():
    """Load the trained segmentation model once and reuse it."""
    global _seg_model
    if _seg_model is None:
        _seg_model = YOLO(SEG_MODEL_PATH)
    return _seg_model


def segment(image, conf: float = 0.25):
    """Run per-tooth segmentation on an image, return the result."""
    model = load_segmenter()
    results = model.predict(image, conf=conf, verbose=False)
    return results[0]