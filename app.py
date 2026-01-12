import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os

# Custom CSS for modern design with animations and transitions
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Modern header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        text-align: center;
        animation: fadeInDown 0.8s ease-out;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: slideInUp 0.8s ease-out;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        animation: fadeIn 1s ease-out 0.3s both;
    }

    /* Card styling for results */
    .prediction-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }

    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    .confidence-high { background: linear-gradient(135deg, #4CAF50, #45a049); }
    .confidence-medium { background: linear-gradient(135deg, #FFC107, #FFA000); }
    .confidence-low { background: linear-gradient(135deg, #F44336, #D32F2F); }

    .confidence-indicator {
        padding: 0.75rem 1.5rem;
        border-radius: 30px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        animation: pulse 2s infinite;
    }

    /* Modern button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-size: 1rem;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }

    .stButton>button:active {
        transform: translateY(1px);
    }

    /* File uploader styling */
    .stFileUploader>div>div>div {
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        background: #f8f9ff;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        animation: fadeIn 0.8s ease-out;
    }

    .stFileUploader>div>div>div:hover {
        border-color: #764ba2;
        background: #f0f4ff;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #667eea;
        border-radius: 10px;
    }

    /* Dataframe table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        animation: fadeIn 0.8s ease-out;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        animation: fadeIn 1s ease-out;
    }

    /* Info cards */
    .info-card {
        background: #f8f9ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
        animation: fadeInRight 0.8s ease-out;
    }

    .info-card:hover {
        transform: translateX(5px);
    }

    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        animation: fadeIn 0.8s ease-out;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .main-header { padding: 1.5rem; }
        .prediction-card, .info-card, .chart-container { padding: 1rem; }
        .confidence-indicator { padding: 0.5rem 1rem; font-size: 1rem; }
    }

    /* Spinner animation */
    .stSpinner {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Mangrove Leaf Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern header
st.markdown('<div class="main-header"><h1>🌿 Mangrove Leaf Classification</h1><p>AI-powered mangrove species identification using ResNet50 CNN</p></div>', unsafe_allow_html=True)

# Add a sidebar-like section for instructions (hidden by default but can be expanded)
with st.expander("ℹ️ Panduan Penggunaan", expanded=False):
    st.markdown("""
    **Cara menggunakan aplikasi ini:**
    1. Upload gambar daun mangrove (format: JPG, JPEG, PNG)
    2. Tunggu proses analisis selesai
    3. Lihat hasil klasifikasi dan tingkat keyakinan model
    4. Gunakan tombol "Unggah Gambar Baru" untuk analisis berikutnya

    **Tips:**
    - Pastikan gambar memiliki pencahayaan yang baik
    - Fokus pada daun mangrove tanpa latar belakang berlebihan
    - Hindari gambar yang terlalu buram atau tidak jelas
    """)

# ==============================
# CONFIG
# ==============================
# Ganti sesuai kelas pada dataset ANDA
classes = ['avicennia', 'bruguiera', 'rhizophora', 'sonneratia']

# Deskripsi tiap kelas (opsional, bisa kamu edit)
mangrove_info = {
    "avicennia": "Genus Avicennia (Api-api) memiliki daun hijau mengkilap dengan ujung meruncing, sering mengeluarkan garam di permukaan daun.",
    "rhizophora": "Rhizophora (Bakau) memiliki daun tebal, hijau cerah, bentuk elips, dan biasanya terdapat bintik hitam kecil.",
    "bruguiera": "Bruguiera (Tancang) daunnya lonjong besar, tebal, hijau gelap, dan terdapat stipula berwarna kemerahan.",
    "sonneratia": "Sonneratia (Pedada) memiliki daun bulat telur, hijau kebiruan dan lebih tipis dibanding mangrove lain."
}

# Referensi gambar (opsional)
ref_images = {
    "avicennia": "https://st2.depositphotos.com/1388500/47887/i/1600/depositphotos_478871782-stock-photo-young-leaves-avicennia-marina-mangrove.jpg",
    "rhizophora": "https://www.nparks.gov.sg/-/media/ffw/migrated/round2/flora/3265/f94a74e8a731401b8d26813c0c51f234.jpg",
    "bruguiera": "https://www.shutterstock.com/image-photo/mangrove-plants-bruguiera-sexangula-260nw-1250844724.jpg",
    "sonneratia": "https://www.shutterstock.com/image-photo/sonneratia-alba-mangrove-tree-family-600w-1505660996.jpg"
}

# ==============================
# CUSTOM MODEL REBUILDING FUNCTION
# ==============================
def rebuild_model_architecture():
    """
    Rebuild the model architecture based on ResNet50 transfer learning
    This matches the architecture used in training with input shape (150, 150, 3)
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, models

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Freeze the base model

    inputs = layers.Input(shape=(150, 150, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)  # Dense layer with 256 units as in training
    x = layers.Dropout(0.5)(x)  # Dropout layer with 0.5 rate as in training
    outputs = layers.Dense(len(classes), activation='softmax')(x)  # 4 classes for mangrove

    model = models.Model(inputs, outputs)
    return model

# ==============================
# LOAD MODEL with maximum error handling
# ==============================
st.subheader("Memuat Model CNN")

@st.cache_resource
def load_cnn_model():
    model_files = [
        "final_daun_resnet50_model.keras",
        "Daun_resnet50_best.keras",
        "final_daun_resnet50_model.h5"
    ]

    for file_path in model_files:
        if os.path.exists(file_path):
            try:
                model = rebuild_model_architecture()
                model.load_weights(file_path)
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                return model
            except:
                pass

            try:
                model = tf.keras.models.load_model(file_path, compile=False)
                return model
            except:
                pass

    return None


st.subheader("Memuat Model CNN")
model = load_cnn_model()

if model is None:
    st.error("❌ Model tidak ditemukan atau gagal dimuat.")
    st.stop()


# Try loading existing models first
model_files = [
    "final_daun_resnet50_model.keras",
    "Daun_resnet50_best.keras",
    "final_daun_resnet50_model.h5"
]

# Try each model file with different parameters
for file_path in model_files:
    if os.path.exists(file_path):
        st.info(f"Mencoba memuat model dari {file_path}")

        # First try building the architecture and loading weights
        try:
            # Build model with the exact architecture used in training
            model = rebuild_model_architecture()

            # Load weights only - this avoids issues with saved model format incompatibilities
            model.load_weights(file_path)

            model_path = file_path
            st.success(f"✅ Model berhasil dibangun dan bobot dimuat dari {file_path}")
            break
        except Exception as e:
            st.warning(f"⚠️ Gagal dengan pendekatan bobot: {str(e)[:100]}...")

        # If the above fails, try with compile=False
        try:
            model = tf.keras.models.load_model(file_path, compile=False)
            model_path = file_path
            st.success(f"✅ Model berhasil dimuat dari {file_path}")
            break
        except Exception as e:
            st.warning(f"⚠️ Gagal dengan compile=False: {str(e)[:100]}...")

        # Try with custom objects if needed
        try:
            custom_objects = {
                'KerasTensor': tf.keras.utils.get_custom_objects().get('KerasTensor'),
            }
            model = tf.keras.models.load_model(file_path, compile=False, custom_objects=custom_objects)
            model_path = file_path
            st.success(f"✅ Model berhasil dimuat dari {file_path} dengan custom_objects")
            break
        except Exception as e:
            st.warning(f"⚠️ Gagal dengan custom_objects: {str(e)[:100]}...")

if model is None:
    st.error("""
    ❌ Tidak ada model yang dapat dimuat dari file yang tersedia.

    Masalah ini terjadi karena versi TensorFlow/Keras yang berbeda atau arsitektur model
    yang kompleks yang tidak dapat dimuat dengan cara standar.

    Untuk memperbaikinya, Anda perlu:
    1. Melatih ulang model dengan versi TensorFlow/Keras yang sesuai
    2. Atau, menyimpan model dengan cara yang lebih kompatibel
    """)
    st.stop()

# Compile the model for predictions
try:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception:
    pass  # If already compiled or cannot compile, continue

# ==============================
# FILE UPLOADER
# ==============================
st.divider()
st.subheader("Upload Gambar Daun Mangrove")

uploaded_file = st.file_uploader(
    "Unggah gambar daun (format: JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICTION PROCESS
# ==============================
if uploaded_file is not None:

    # Create two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Prediction card
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

        # Preprocessing gambar sesuai input ResNet50 - menggunakan ukuran yang sama dengan pelatihan (150,150)
        img = image.load_img(uploaded_file, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Model ResNet50 expects input normalized in a specific way
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        with st.spinner("🔄 Memproses gambar..."):
            try:
                prediction = model.predict(img_array)
                predicted_class = classes[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
            except Exception as e:
                st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
                st.stop()

        # ==============================
        # OUTPUT HASIL PREDIKSI
        # ==============================
        st.subheader("🌱 Hasil Prediksi")
        st.success(f"Jenis mangrove terdeteksi: **{predicted_class.capitalize()}**")

        # Confidence indicator with dynamic class
        if confidence > 90:
            conf_class = "confidence-high"
            conf_text = "Model sangat yakin dengan prediksi ini."
        elif confidence > 70:
            conf_class = "confidence-medium"
            conf_text = "Model cukup yakin — verifikasi dengan gambar lain untuk kepastian."
        else:
            conf_class = "confidence-low"
            conf_text = "Model masih ragu. Gunakan gambar daun yang lebih jelas."

        st.markdown(f'<div class="confidence-indicator {conf_class}">Tingkat keyakinan: {confidence:.2f}%</div>', unsafe_allow_html=True)
        st.info(conf_text)

        # Info tambahan
        if predicted_class in mangrove_info:
            st.markdown(f"**Deskripsi:** {mangrove_info[predicted_class]}", unsafe_allow_html=True)

        # Referensi gambar
        if predicted_class in ref_images:
            st.image(ref_images[predicted_class],
                     caption=f"Contoh daun {predicted_class.capitalize()}",
                     use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # SHOW CLASS PROBABILITY TABLE
    # ==============================
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Probabilitas Tiap Kelas")

    prob_df = pd.DataFrame({
        "Kelas": [cls.capitalize() for cls in classes],
        "Probabilitas (%)": [round(p * 100, 2) for p in prediction[0]]
    }).sort_values(by="Probabilitas (%)", ascending=False)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(prob_df, use_container_width=True)
    with col2:
        st.bar_chart(prob_df.set_index("Kelas"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    if st.button("🔄 Unggah Gambar Baru"):
        st.rerun()

else:
    # Show info cards when no file is uploaded
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**🎯 Fungsi Utama**")
        st.markdown("Aplikasi ini menggunakan CNN berbasis ResNet50 untuk mengklasifikasikan jenis mangrove berdasarkan gambar daun.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**📋 Kelas yang Didukung**")
        st.markdown("- Avicennia (Api-api)\n- Bruguiera (Tancang)\n- Rhizophora (Bakau)\n- Sonneratia (Pedada)")
        st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
**Dibuat oleh:** DIMAS KURNIAWAN dan Putri Sapela Loka sari
**Teknologi:** TensorFlow + Streamlit | **Model:** ResNet50 – Klasifikasi Daun Mangrove
""")
st.markdown('</div>', unsafe_allow_html=True)

# Add a final animation trigger to ensure all elements are properly animated
st.markdown('<script>document.addEventListener("DOMContentLoaded", function(){ window.dispatchEvent(new Event("resize")); });</script>', unsafe_allow_html=True)