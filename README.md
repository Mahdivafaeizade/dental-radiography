# Dental X-ray AI

Multi-stage dental-AI project: classification, object detection, and (planned) segmentation of dental X-ray findings (Cavity, Fillings, Impacted Tooth, Implant).

## Roadmap

1. **Stage 1 (done)** — image classification: does this X-ray contain each finding? (multi-label, since most X-rays have several findings at once)
2. **Stage 2 (done)** — object detection: exactly *where* is each finding located (bounding boxes), using YOLO11
3. **Stage 3 (planned)** — segmentation (pixel-level tooth boundaries) using a second dataset + a single unified UI combining all three models

## Problem

Manually reviewing dental X-rays for common findings is time-consuming. This model gives a fast first-pass screen: upload an X-ray, get a probability for each of 4 common finding types.

## Data

[Dental Radiography](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography) — 1075 train / 121 valid / 73 test X-ray images. The dataset ships with bounding-box annotations (built for object detection), not classification labels — this project derives multi-label targets by collapsing each image's set of bounding-box classes into a 4-way binary vector (`Cavity`, `Fillings`, `Impacted Tooth`, `Implant`).

**Why multi-label, not single-label:** EDA showed 62% of images (665/1075) contain more than one finding type — a single "dominant label" approach would discard real information for most of the dataset.

## Approach

### Stage 1: Classification

1. **EDA** — confirmed the multi-label nature of the data and measured class imbalance (Fillings present in 80% of images, Cavity in only 21%).
2. **Model** — MobileNetV2 (ImageNet-pretrained), final layer replaced for 4-way multi-label output.
3. **Training** — `BCEWithLogitsLoss` with `pos_weight` to counteract class imbalance (without it, the model would learn to just predict "no" for rare classes like Cavity). Best-checkpoint saving based on validation loss, since training loss kept improving past the point where validation loss started getting worse (overfitting after ~epoch 3).
4. **Evaluation** — per-class precision/recall/F1 (not just loss), since loss alone doesn't reveal practical performance on rare classes.

### Stage 2: Object detection

1. **Conversion** — the dataset's bounding-box CSV annotations (pixel corner coordinates) converted to YOLO format (normalized center coordinates + per-image label files).
2. **Model** — YOLO11n (nano), transfer-learned from COCO pretrained weights.
3. **Training** — first trained on CPU (30 epochs, 416px, ~57 min) as a baseline, then retrained on GPU (RTX 3050 Ti) at full 640px resolution for up to 100 epochs with early stopping (patience=20) — GPU training was ~5x faster per epoch, and the larger image size + more epochs meaningfully improved detection of small/subtle findings.

## Results

### Stage 1: Classification (precision / recall / F1)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fillings | 0.88 | 0.72 | 0.79 |
| Implant | 0.70 | 0.75 | 0.72 |
| Impacted Tooth | 0.42 | 0.60 | 0.49 |
| Cavity | 0.37 | 0.61 | 0.46 |

Rare classes (Cavity, Impacted Tooth) show higher recall than precision — the model errs toward flagging possible findings rather than missing them, a reasonable tradeoff for a screening tool.

### Stage 2: Object detection (mAP50)

| Class | CPU baseline (416px, 30 epochs) | GPU (640px, 53 epochs) |
|---|---|---|
| Cavity | 0.206 | **0.374** |
| Fillings | 0.858 | 0.904 |
| Impacted Tooth | 0.791 | 0.808 |
| Implant | 0.961 | 0.970 |
| **Overall** | 0.704 | **0.764** |

GPU retraining nearly doubled Cavity detection (our weakest class) and improved every other class too. Inference dropped from 14.3ms/image (CPU) to 3.2ms/image (GPU).

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography) into `data/` (train/valid/test folders + `_annotations.csv` each).

## Usage

Train the classifier (stage 1):
```bash
python3 -m src.train
python3 -m src.evaluate
```

Convert annotations and train the detector (stage 2):
```bash
python3 -m src.yolo_convert
python3 -m src.train_yolo
```

Run the UI (both stages, tabbed):
```bash
streamlit run app.py
```

Run the classification API:
```bash
python3 -m uvicorn src.api:app --reload
```

## Tech stack

Python, PyTorch, torchvision (MobileNetV2 transfer learning), Ultralytics YOLO11 (object detection), FastAPI, Streamlit.
