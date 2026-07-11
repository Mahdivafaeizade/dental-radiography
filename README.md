# Dental X-ray Classifier

Multi-label classification of dental X-ray findings (Cavity, Fillings, Impacted Tooth, Implant) using transfer learning. Stage 1 of a planned 3-stage dental-AI project.

## Roadmap

1. **Stage 1 (this repo)** — image classification: does this X-ray contain each finding? (multi-label, since most X-rays have several findings at once)
2. **Stage 2 (planned)** — object detection on the same dataset: exactly *where* is each finding located (bounding boxes)
3. **Stage 3 (planned)** — segmentation (pixel-level tooth boundaries) + a single unified UI combining all three models

## Problem

Manually reviewing dental X-rays for common findings is time-consuming. This model gives a fast first-pass screen: upload an X-ray, get a probability for each of 4 common finding types.

## Data

[Dental Radiography](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography) — 1075 train / 121 valid / 73 test X-ray images. The dataset ships with bounding-box annotations (built for object detection), not classification labels — this project derives multi-label targets by collapsing each image's set of bounding-box classes into a 4-way binary vector (`Cavity`, `Fillings`, `Impacted Tooth`, `Implant`).

**Why multi-label, not single-label:** EDA showed 62% of images (665/1075) contain more than one finding type — a single "dominant label" approach would discard real information for most of the dataset.

## Approach

1. **EDA** — confirmed the multi-label nature of the data and measured class imbalance (Fillings present in 80% of images, Cavity in only 21%).
2. **Model** — MobileNetV2 (ImageNet-pretrained), final layer replaced for 4-way multi-label output. Chosen for CPU-friendly training speed given no working local GPU.
3. **Training** — `BCEWithLogitsLoss` with `pos_weight` to counteract class imbalance (without it, the model would learn to just predict "no" for rare classes like Cavity). Best-checkpoint saving based on validation loss, since training loss kept improving past the point where validation loss started getting worse (overfitting after ~epoch 3).
4. **Evaluation** — per-class precision/recall/F1 (not just loss), since loss alone doesn't reveal practical performance on rare classes.

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fillings | 0.88 | 0.72 | 0.79 |
| Implant | 0.70 | 0.75 | 0.72 |
| Impacted Tooth | 0.42 | 0.60 | 0.49 |
| Cavity | 0.37 | 0.61 | 0.46 |

Rare classes (Cavity, Impacted Tooth) show higher recall than precision — the model errs toward flagging possible findings rather than missing them, a reasonable tradeoff for a screening tool.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography) into `data/` (train/valid/test folders + `_annotations.csv` each).

## Usage

Train:
```bash
python3 -m src.train
```

Evaluate:
```bash
python3 -m src.evaluate
```

Run the UI:
```bash
streamlit run app.py
```

Run the API:
```bash
python3 -m uvicorn src.api:app --reload
```

## Tech stack

Python, PyTorch, torchvision (MobileNetV2 transfer learning), FastAPI, Streamlit.
