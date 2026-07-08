import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load CNN Model
# -----------------------------
model = load_model("model/cnn_model.keras")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📌 About")

    st.markdown("""
### ✍️ Handwritten Digit Recognition

This application uses a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** to recognize handwritten digits.

### 🛠 Technologies
- ✔ TensorFlow / Keras
- ✔ Python
- ✔ NumPy
- ✔ Pillow
- ✔ Streamlit

### 📊 Model
- CNN Architecture
- MNIST Dataset
- Test Accuracy: **99.09%**

---
**CodeAlpha Machine Learning Internship**

**Developed by:**  
**B.N. Sandhya**
""")

# -----------------------------
# Main Title
# -----------------------------
st.title("✍️ Handwritten Digit Recognition System")

st.write(
    "Upload a handwritten digit image (0–9). The trained CNN model will predict the digit and display its confidence score."
)

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    img = np.array(image)

    # Invert colors if background is white
    if img.mean() > 127:
        img = 255 - img

    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    st.image(image, caption="Uploaded Image", width=250)

    prediction = model.predict(img, verbose=0)

    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"🎯 Predicted Digit: {digit}")

    st.info(f"📊 Confidence: {confidence:.2f}%")

    st.progress(int(confidence))

    st.subheader("Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"Digit {i}: {prob*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2026 B.N. Sandhya | CodeAlpha Machine Learning Internship")
