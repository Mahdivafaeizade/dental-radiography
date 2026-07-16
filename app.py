"""Streamlit UI for the dental X-ray classifier, detector, and segmenter."""

import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data_loader import CLASSES
from src.dataset import EVAL_TRANSFORMS
from src.detect import load_detector
from src.segment import load_segmenter
from src.train import build_model, DEVICE

st.set_page_config(page_title="Dental X-ray AI", page_icon="🦷")
st.title("🦷 Dental X-ray AI")
st.caption("Stage 1: classification · Stage 2: detection (where) · Stage 3: segmentation (which tooth)")


@st.cache_resource
def load_classifier():
    model = build_model()
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
    model.eval()
    return model


classifier = load_classifier()
detector = load_detector()
segmenter = load_segmenter()

uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tab1, tab2, tab3 = st.tabs(
        ["Stage 1: Classification", "Stage 2: Detection", "Stage 3: Segmentation"]
    )

    with tab1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        tensor = EVAL_TRANSFORMS(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = classifier(tensor)
            probs = torch.sigmoid(outputs).squeeze(0).cpu().tolist()

        st.subheader("Predictions")
        for cls, prob in zip(CLASSES, probs):
            st.progress(prob, text=f"{cls}: {prob:.1%}")

    with tab2:
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
        results = detector.predict(np.array(image), conf=conf_threshold, verbose=False)[0]
        annotated = results.plot()[..., ::-1]
        st.image(annotated, caption="Detected findings", use_container_width=True)

        st.subheader("Detections")
        if len(results.boxes) == 0:
            st.write("No findings detected above this confidence threshold.")
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"- **{results.names[cls_id]}**: {conf:.1%} confidence")

    with tab3:
        seg_conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05, key="seg_conf")
        seg_results = segmenter.predict(np.array(image), conf=seg_conf, verbose=False)[0]
        seg_annotated = seg_results.plot(boxes=False, labels=False, conf=False)[..., ::-1]
        st.image(seg_annotated, caption="Segmented teeth (Universal numbering)", use_container_width=True)

        st.subheader("Teeth identified")
        if seg_results.boxes is None or len(seg_results.boxes) == 0:
            st.write("No teeth segmented above this confidence threshold.")
        else:
            teeth = sorted(
                {seg_results.names[int(b.cls[0])] for b in seg_results.boxes},
                key=lambda x: int(x),
            )
            st.write(f"Tooth numbers: {', '.join(teeth)}")