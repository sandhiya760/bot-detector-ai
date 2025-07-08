# 🤖 AI Bot Detector using Machine Learning & Streamlit

This project is an AI-powered web application designed to detect **bot-like behavior** based on user interaction data. It uses a **Multilayer Perceptron (MLPClassifier)** for classification and **Streamlit** for a fast, interactive web interface.

---

## 📌 Features

- ✅ Detects bots using behavioral data (e.g., click speed, scroll ratio, hover time)
- 📈 Machine Learning-based model (MLP Classifier)
- 🧠 Trained on synthetic + customizable datasets
- 🖥️ Streamlit-powered interface
- 📊 Real-time prediction with clean UI

---

## 🛠️ Technologies Used

| Component          | Tech                         |
|-------------------|------------------------------|
| Language          | Python 3.13                  |
| ML Model          | MLPClassifier (Scikit-learn) |
| Frontend UI       | Streamlit                    |
| Dataset Format    | CSV                          |
| Hosting/Versioning| GitHub                       |

---

## 📂 Project Structure

bot-detector-ai/
│
├── app.py # Streamlit app: handles user input & shows prediction
├── bot_detector.py # Script to train and save the ML model
├── bot_model.pkl # Saved trained model (MLPClassifier)
├── simulated_bot_data.csv # Sample dataset with labeled behavior data
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation (you're reading it!)

## 📊 Dataset

A simulated dataset with the following behavioral features:

- `clicks_per_minute`
- `time_on_site` (in seconds)
- `page_views`

Label:
- `0` → Human  
- `1` → Bot

> In future iterations, more nuanced features like `scroll_ratio`, `hover_time`, `form_fill_ratio`, and `mouse_entropy` can be added.

---

## 🧠 Machine Learning Approach

- Model used: **MLPClassifier** (Multi-Layer Perceptron Neural Network)
- Reason: Ideal for detecting **non-linear patterns** in small datasets
- Dataset split into training and testing sets
- Evaluated using classification metrics like **precision, recall, and f1-score**
- Model serialized using `joblib` and saved as `bot_model.pkl`

---

## 🌐 Streamlit Web App

A simple and responsive UI to test input behavior manually.

### 🔹 Features:
- Input form for behavioral values
- Displays result: ✅ **"Human Verified"** or 🤖 **"Bot Detected"**
- Lightweight & deployable on Render, Hugging Face Spaces, or localhost

---

### ▶️ To Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
🔮 Future Scope
Include advanced behavioral metrics like:

Mouse movement entropy

Scroll ratio

Hover time

CAPTCHA response time

Deploy web app online

Add bot heatmap visualization and graph insights

📚 Lessons Learned
Simulating and preprocessing bot-human behavioral datasets

Building ML classification pipelines using MLP

Saving/loading models using joblib

Creating UI with Streamlit

Managing and pushing projects on GitHub

👩‍💻 Author
Sandhiya Srinivasan
Final Year ECE Student
Cybersecurity | AI | Embedded Systems
📫 sandhiyasrinivasan0620005@gmail.com

