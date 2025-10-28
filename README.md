# 🌍 Landmark Recognition App

هذا التطبيق يستخدم **الذكاء الاصطناعي** للتعرف على المعالم السياحية والصور الخاصة بها في أفريقيا، ويعرض الموقع على الخريطة بشكل احترافي مع رابط مباشر لـ Google Maps.

---

## **مميزات التطبيق**

- التعرف على المعالم من الصور باستخدام **TensorFlow Hub**  
- عرض الـ **Address** بشكل منظم وسطر بسطر  
- عرض الموقع على **خريطة تفاعلية** مع رابط مباشر لـ Google Maps  
- واجهة **Streamlit حديثة واحترافية**  
- رفع الصور مباشر وتجربة التنبؤ في الوقت الفعلي  

---

# Landmark Recognition

## Model Download
Download the trained model from [this link](https://landmarks-gd39rndyppre6zr2jde2cw.streamlit.app/k).

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run LM.py`

---

## **متطلبات التشغيل**

- Python 3.9 أو أحدث  
- المكتبات المطلوبة:

```bash
pip install streamlit tensorflow tensorflow-hub pandas numpy pillow geopy
