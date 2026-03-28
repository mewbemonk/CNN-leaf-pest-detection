import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import streamlit as st
import os

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = 128
MODEL_PATH = "new_plant_model.h5"

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌾 फसल रोग पहचान प्रणाली")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file नहीं मिला!")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# CLASS LABELS (FULL LIST ✅)
# ==============================
class_labels = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry___Powdery_mildew",
"Cherry___healthy",
"Corn___Cercospora_leaf_spot Gray_leaf_spot",
"Corn___Common_rust",
"Corn___Northern_Leaf_Blight",
"Corn___healthy",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites",
"Tomato___Target_Spot",
"Tomato___Yellow_Leaf_Curl_Virus",
"Tomato___mosaic_virus",
"Tomato___healthy"
]

# ==============================
# DOWNLOAD BUTTON
# ==============================
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        st.download_button("📥 मॉडल डाउनलोड करें", f, file_name="plant_model.h5")

# ==============================
# IMAGE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📷 पत्ती की फोटो अपलोड करें", type=["jpg","png","jpeg"])

# ==============================
# PREDICTION
# ==============================
if uploaded_file is not None and model is not None:

    st.image(uploaded_file, caption="अपलोड की गई फोटो")

    img = load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    # 🔥 SAFE FIX (no crash)
    if class_index < len(class_labels):
        result = class_labels[class_index]
    else:
        result = "Unknown"

    clean_result = result.replace("___"," - ").replace("_"," ")

    st.success(f"🌿 बीमारी: {clean_result}")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")

    st.markdown("---")
    st.subheader("👨‍🌾 सलाह:")

    result_lower = result.lower()

    if "healthy" in result_lower:
        st.success("✅ पौधा स्वस्थ है")
    elif "scab" in result_lower:
        st.warning("💊 Captan spray करें")
    elif "rust" in result_lower:
        st.warning("💊 Fungicide use करें")
    elif "blight" in result_lower:
        st.warning("💊 Mancozeb spray करें")
    elif "aphid" in result_lower or "mite" in result_lower:
        st.warning("🐛 Neem oil spray करें")
    else:
        st.info("कृषि विशेषज्ञ से संपर्क करें")
