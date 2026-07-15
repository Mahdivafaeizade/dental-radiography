"""Streamlit UI for the dental X-ray classifier (stage 1) and detector (stage 2)."""

import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data_loader import CLASSES
from src.dataset import EVAL_TRANSFORMS
from src.detect import load_detector
from src.train import build_model, DEVICE

st.set_page_config(page_title="Dental X-ray AI", page_icon="🦷")
st.title("🦷 Dental X-ray AI")
st.caption("Stage 1: classification (what findings are present) · Stage 2: detection (exactly where)")


@st.cache_resource
def load_classifier():
    model = build_model()
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
    model.eval()
    return model


classifier = load_classifier()
detector = load_detector()

uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tab1, tab2 = st.tabs(["Stage 1: Classification", "Stage 2: Detection"])

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