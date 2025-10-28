# app.py
import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from io import BytesIO
import requests

# -----------------------------
# ستايل CSS
# -----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    .stFileUploader>div {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
    }
    .stImage img {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .card {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
    }
    .gm-btn {
        display:inline-block;
        background-color:#4285F4;
        color:white;
        padding:10px 14px;
        border-radius:8px;
        text-decoration:none;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# CSV Labels من GitHub (Raw link)
# -----------------------------
labels_csv_url = "https://raw.githubusercontent.com/Karim-Anwr/Landmarks/main/landmarks_label.csv"
df_labels = pd.read_csv(labels_csv_url)
labels = dict(zip(df_labels.id, df_labels.name))

# -----------------------------
# موديل TensorFlow Hub
# -----------------------------
model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_africa_V1/1'
classifier_layer = hub.KerasLayer(model_url, input_shape=(321, 321, 3), output_key="predictions:logits")

# -----------------------------
# دالة معالجة الصورة والتنبؤ
# -----------------------------
def image_processing(image_data):
    img_shape = (321, 321)
    img = Image.open(image_data).convert("RGB")
    img_resized = img.resize(img_shape)
    img_array = np.array(img_resized) / 255.0
    img_array = img_array[np.newaxis, ...]
    result = classifier_layer(img_array)
    predicted_id = int(np.argmax(result.numpy()))
    predicted_label = labels.get(predicted_id, "Unknown")
    return predicted_label, img_resized

# -----------------------------
# جلب عنوان وإحداثيات
# -----------------------------
def get_map(location_name):
    geolocator = Nominatim(user_agent="LandmarkApp")
    try:
        location = geolocator.geocode(location_name, timeout=10)
    except:
        return None, None, None
    if location:
        return location.address, location.latitude, location.longitude
    return None, None, None

# -----------------------------
# عرض Address منظم
# -----------------------------
def display_address(address):
    parts = [p.strip() for p in address.split(',') if p.strip()]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🏠 Address Details")
    for part in parts:
        st.markdown(f"<div style='margin-bottom:6px'>• {part}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# عرض الخريطة مع رابط Google Maps
# -----------------------------
def display_map_and_link(lat, lon, place_name):
    df_map = pd.DataFrame([[lat, lon]], columns=['lat', 'lon'])
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Location")
    st.map(df_map, use_container_width=True)
    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    st.markdown(
        f"""
        <div style="margin-top:10px">
            <a class="gm-btn" href="{gmaps_url}" target="_blank" rel="noopener">
                افتح الموقع في Google Maps — {place_name}
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# الواجهة الرئيسية
# -----------------------------
def run():
    st.set_page_config(page_title="🌍 Landmark Recognition", layout="wide")

    # Header مع شعار من رابط مباشر
    logo_url = "https://raw.githubusercontent.com/Karim-Anwr/Landmarks/main/logo.png"
    st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
    try:
        logo_img = Image.open(requests.get(logo_url, stream=True).raw).resize((56, 56))
        st.image(logo_img, use_container_width=False)
    except:
        st.write("Logo not found")
    st.markdown("<h1 style='margin:0'>🌍 Landmark Recognition</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("اكتشف المعالم باستخدام AI — ارفع صورة وشوف النتيجة")
    st.markdown("---")

    img_file = st.file_uploader("اختر صورتك", type=['png','jpg','jpeg'])
    if img_file is not None:
        img_bytes = BytesIO(img_file.read())
        prediction, processed_img = image_processing(img_bytes)

        col1, col2 = st.columns([1,1])
        with col1:
            st.image(processed_img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📍 Predicted Landmark")
            st.markdown(f"<p style='font-size:18px;font-weight:bold;color:#2E7D32;margin:6px 0'>{prediction}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            address, lat, lon = get_map(prediction)
            if address and lat is not None and lon is not None:
                display_address(address)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 🌐 Coordinates")
                st.markdown(f"- **Latitude:** {lat:.6f}  \n- **Longitude:** {lon:.6f}", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                display_map_and_link(lat, lon, prediction)
            else:
                st.warning("No address found for this landmark (أو فشل الاتصال بـ geocoder).")

if __name__ == "__main__":
    run()
