# ✍️ Handwritten Digit Recognition System

A deep learning web application developed using **TensorFlow**, **Keras**, and **Streamlit** that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN).

---

# 📌 Project Overview

This project was developed as part of the **CodeAlpha Machine Learning Internship**.

The application allows users to upload an image containing a handwritten digit and predicts the digit with a confidence score.

---

# 🚀 Features

* Upload handwritten digit images
* CNN-based digit recognition
* Displays predicted digit
* Displays confidence score
* Shows prediction probabilities for digits 0–9
* Professional Streamlit interface
* Sidebar with project information

---

# 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pillow
* Streamlit

---

# 📂 Project Structure

```text
CodeAlpha_Handwritten_Recognition/
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
├── images/
├── model/
│   └── cnn_model.keras
└── screenshots/
```

---

# 🧠 Model Information

* Model Type: Convolutional Neural Network (CNN)
* Dataset: MNIST
* Test Accuracy: **99.09%**

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CodeAlpha_Handwritten_Recognition.git
```

Move into the project folder

```bash
cd CodeAlpha_Handwritten_Recognition
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📷 Application Preview

Upload a handwritten digit image.

The application displays:

* Uploaded Image
* Predicted Digit
* Confidence Percentage
* Prediction Probabilities

---

# 📸 Screenshots

Add screenshots of the application inside the `screenshots/` folder.

Example:

* Home Page
* Uploaded Image
* Prediction Result

---

# 👨‍💻 Developed By

**B.N. Sandhya**

Machine Learning Intern

CodeAlpha Internship

---

# ⭐ Future Improvements

* Support handwritten letters (A–Z)
* Predict complete handwritten words
* Improve image preprocessing
* Deploy online using Streamlit Community Cloud
