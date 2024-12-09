import streamlit as st
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image

import sys
print(sys.path)
print(tf.__file__)

# Judul Aplikasi
st.set_page_config(page_title="CNN Model Predictor", page_icon=":camera:")

# Muat model CNN yang sudah dilatih
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('C:/Users/andre/Downloads/model_96.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fungsi preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Resize gambar
    img = image.resize(target_size)
    
    # Konversi ke array NumPy dan normalisasi
    img_array = np.array(img) / 255.0
    
    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Fungsi prediksi
def predict_image(model, image):
    # Preprocessing gambar
    processed_image = preprocess_image(image)
    
    # Lakukan prediksi
    prediction = model.predict(processed_image)
    print(prediction)
    
    # Definisikan label kelas (sesuaikan dengan model Anda)
    class_labels = ['bottlecap', 'cans', 'cardboard', 'ceramicsbowl', 'disc', 'galvanizedsteel', 'glassbottle', 'newspaper', 'paper', 'pen', 'plasticbag', 'plasticbottle', 'rag', 'spoonfork', 'tire', 'watergallon']
    
    # Dapatkan kelas dengan probabilitas tertinggi
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)] * 100
    
    return predicted_class, confidence

# Judul Aplikasi
st.title("🖼️ CNN Model Predictor")
st.write("Unggah gambar atau gunakan kamera untuk prediksi")

# Muat model
model = load_model()

if model is not None:
    # Pilih sumber input
    input_source = st.radio("Pilih sumber gambar:", 
                            ["Unggah Gambar", "Ambil Foto dari Kamera"])

    if input_source == "Unggah Gambar":
        # Upload gambar
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        
        if uploaded_file is not None:
            # Tampilkan gambar yang diunggah
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)
            
            # Tombol prediksi
            if st.button("Prediksi Gambar"):
                # Lakukan prediksi
                predicted_class, confidence = predict_image(model, image)
                
                # Tampilkan hasil
                st.success(f"Prediksi: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")

    else:
        # Ambil foto dari kamera
        camera_image = st.camera_input("Ambil Foto")
        
        if camera_image is not None:
            # Konversi input kamera ke PIL Image
            image = Image.open(camera_image)
            st.image(image, caption="Foto yang diambil", use_column_width=True)
            
            # Tombol prediksi
            if st.button("Prediksi Foto"):
                # Lakukan prediksi
                predicted_class, confidence = predict_image(model, image)
                
                # Tampilkan hasil
                st.success(f"Prediksi: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")

else:
    st.error("Gagal memuat model. Pastikan file model benar.")

# Footer
st.markdown("---")
st.markdown("Aplikasi Prediksi Model CNN")