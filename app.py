import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Brain Scan Analyzer",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Custom CSS (UI Styling)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00D4FF;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header Section
# =========================
st.markdown('<div class="title">🧠 Brain Scan Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered MRI & CT Scan Detection</div>', unsafe_allow_html=True)
st.write("")

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    mri = tf.keras.models.load_model("hestnet_mri_model.keras", compile=False)
    ct = tf.keras.models.load_model("ds_vit_ct_model.keras", compile=False)
    return mri, ct

mri_model, ct_model = load_models()

MRI_CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']

# =========================
# Preprocessing
# =========================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# Modality Detection
# =========================
def detect_modality(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges) / (224 * 224)

    return "CT" if edge_ratio > 5 else "MRI"

# =========================
# Prediction Functions
# =========================
def predict_mri(img):
    pred = mri_model.predict(img)
    idx = np.argmax(pred)
    return MRI_CLASSES[idx], np.max(pred)

def predict_ct(img):
    pred = ct_model.predict(img)
    prob = pred[0][0]
    return ("Tumor", prob) if prob > 0.5 else ("No Tumor", 1 - prob)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Upload brain scan images")
    st.info("Supports MRI & CT scans")
    st.write("---")
    st.write("👨‍💻 Developed for Medical AI Project")

# =========================
# Upload Section
# =========================
uploaded_files = st.file_uploader(
    "📤 Upload Brain Scan Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# =========================
# Processing Section
# =========================
if uploaded_files:

    cols = st.columns(2)

    for i, file in enumerate(uploaded_files):

        image = Image.open(file).convert("RGB")
        img = preprocess_image(image)

        modality = detect_modality(image)

        if modality == "MRI":
            label, confidence = predict_mri(img)
            color = "🟢" if label == "notumor" else "🔴"
        else:
            label, confidence = predict_ct(img)
            color = "🟢" if label == "No Tumor" else "🔴"

        with cols[i % 2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.image(image, use_container_width=True)

            st.markdown(f"### 🧾 Type: `{modality}`")
            st.markdown(f"### 🧬 Result: {color} `{label}`")

            st.progress(float(confidence))
            st.write(f"Confidence: **{confidence:.2f}**")

            st.markdown('</div>', unsafe_allow_html=True)