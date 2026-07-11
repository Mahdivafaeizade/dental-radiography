"""Streamlit UI for the dental X-ray multi-label classifier."""

import io

import streamlit as st
import torch
from PIL import Image

from src.data_loader import CLASSES
from src.dataset import EVAL_TRANSFORMS
from src.train import build_model, DEVICE

st.set_page_config(page_title="Dental X-ray Classifier", page_icon="🦷")
st.title("🦷 Dental X-ray Classifier")
st.caption("Stage 1 of 3: multi-label classification. Upload a dental X-ray to check for common findings.")


@st.cache_resource
def load_model():
    model = build_model()
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
    model.eval()
    return model


model = load_model()

uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    tensor = EVAL_TRANSFORMS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().tolist()

    st.subheader("Predictions")
    for cls, prob in zip(CLASSES, probs):
        st.progress(prob, text=f"{cls}: {prob:.1%}")