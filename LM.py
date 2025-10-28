import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import os

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
# إعداد المسارات وملف اللابلز
# -----------------------------
labels_csv_path = r'D:\Computer Vision\LandMarks\landmarks_label.csv'
if not os.path.exists(labels_csv_path):
    st.error(f"Labels file not found at {labels_csv_path}")
    st.stop()

df_labels = pd.read_csv(labels_csv_path)
labels = dict(zip(df_labels.id, df_labels.name))

# -----------------------------
# تحميل موديل TensorFlow Hub مرة واحدة
# -----------------------------
model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_africa_V1/1'
# نستخدم output_key عشان الـ KerasLayer يرجع logits مباشرة
classifier_layer = hub.KerasLayer(model_url, input_shape=(321, 321, 3), output_key="predictions:logits")


# -----------------------------
# دالة معالجة الصورة والتنبؤ
# -----------------------------
def image_processing(image_path):
    img_shape = (321, 321)
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(img_shape)
    img_array = np.array(img_resized) / 255.0
    img_array = img_array[np.newaxis, ...]  # (1,321,321,3)
    result = classifier_layer(img_array)  # يرجع tensor للـ logits
    predicted_id = int(np.argmax(result.numpy()))
    predicted_label = labels.get(predicted_id, "Unknown")
    return predicted_label, img_resized


# -----------------------------
# دالة الحصول على عنوان واحداثيات من Nominatim
# -----------------------------
def get_map(location_name):
    geolocator = Nominatim(user_agent="LandmarkApp")
    try:
        location = geolocator.geocode(location_name, timeout=10)
    except Exception as e:
        return None, None, None
    if location:
        return location.address, location.latitude, location.longitude
    return None, None, None


# -----------------------------
# دالة لعرض Address منظم سطر بسطر
# -----------------------------
def display_address(address):
    parts = [p.strip() for p in address.split(',') if p.strip()]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🏠 Address Details")
    for i, part in enumerate(parts):
        st.markdown(f"<div style='margin-bottom:6px'>• {part}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# دالة عرض الخريطة البسيطة مع رابط Google Maps
# -----------------------------
def display_map_and_link(lat, lon, place_name):
    # DataFrame للخريطة البسيطة (streamlit map)
    df_map = pd.DataFrame([[lat, lon]], columns=['lat', 'lon'])

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Location")
    # عرض الخريطة الافتراضية كنسخة احتياطية (ستركيت بسيطة وسريعة)
    st.map(df_map, use_container_width=True)

    # رابط Google Maps
    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    # رابط جوجل مابس مع اسم المكان (يفتح في نافذة جديدة عند الضغط)
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
    # Header مع شعار صغير
    logo_path = r'D:\Computer Vision\LandMarks\logo.png'
    st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path).resize((56, 56))
        st.image(logo_img, use_container_width=False)
    st.markdown("<h1 style='margin:0'>🌍 Landmark Recognition</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("اكتشف المعالم باستخدام AI — ارفع صورة وشوف النتيجة")
    st.markdown("---")

    img_file = st.file_uploader("اختر صورتك", type=['png', 'jpg', 'jpeg'])
    if img_file is not None:
        upload_dir = r'D:\Computer Vision\LandMarks\Uploaded_Images'
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, img_file.name)
        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        # التنبؤ
        prediction, processed_img = image_processing(save_path)

        # عرض الصورة والتنبؤ في عمودين
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(processed_img, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📍 Predicted Landmark")
            st.markdown(f"<p style='font-size:18px;font-weight:bold;color:#2E7D32;margin:6px 0'>{prediction}</p>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # جلب العنوان والـ coords
            address, lat, lon = get_map(prediction)
            if address and lat is not None and lon is not None:
                display_address(address)
                # نعرض الاحداثيات بشكل بسيط (سطر واحد، من غير JSON)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 🌐 Coordinates")
                st.markdown(f"- **Latitude:** {lat:.6f}  \n- **Longitude:** {lon:.6f}", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # عرض الخريطة + رابط Google Maps
                display_map_and_link(lat, lon, prediction)
            else:
                st.warning("No address found for this landmark (أو فشل الاتصال بـ geocoder).")


if __name__ == "__main__":
    run()
