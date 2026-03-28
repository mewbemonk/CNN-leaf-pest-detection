import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import os
import streamlit as st

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 3   # increase later for better accuracy
MODEL_PATH = "new_plant_model.h5"
DATASET_PATH = "dataset/Plant_leave_diseases_dataset_without_augmentation"

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌾 फसल रोग पहचान प्रणाली")

# ==============================
# LOAD DATA
# ==============================
@st.cache_resource
def load_data():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    return train, val

train_data, val_data = load_data()
class_labels = list(train_data.class_indices.keys())
num_classes = len(class_labels)

# ==============================
# LOAD / TRAIN MODEL
# ==============================
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)

    st.warning("⚡ Model नहीं मिला → Training शुरू हो रही है...")

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    with st.spinner("🚀 Model training चल रही है..."):
        model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

    model.save(MODEL_PATH)
    st.success("✅ Model train होकर save हो गया!")

    return model

model = load_or_train_model()

# ==============================
# DOWNLOAD BUTTON
# ==============================
with open(MODEL_PATH, "rb") as f:
    st.download_button("📥 मॉडल डाउनलोड करें", f, file_name="plant_model.h5")

# ==============================
# IMAGE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📷 पत्ती की फोटो अपलोड करें", type=["jpg","png","jpeg"])

# ==============================
# PREDICTION
# ==============================
if uploaded_file is not None:
    st.image(uploaded_file, caption="अपलोड की गई फोटो")

    img = load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    result = class_labels[class_index]
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