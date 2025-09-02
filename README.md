# 🌾 KrishiMitra AI

AI-powered agricultural assistant that helps farmers with **crop yield prediction**, farming tips, and real-time agricultural insights using **LangChain**, **Groq LLM**, and **HuggingFace models**.

---

## 🚀 Features
- 📊 **Crop Yield Prediction** — Predicts crop yield using **RandomForest Classifier** trained on historical agricultural data (rainfall, fertilizer, pesticide, crop, area, and season).
- 🤖 **AI Chatbot** — Provides farming advice, explains predictions, and answers farmer queries in natural language.
- 🔍 **HuggingFace Integration** — Uses pre-trained embeddings for better recommendations.
- ⚡ **Groq LLM** — Ultra-fast inference with Groq’s high-speed LLMs.
- 🌱 **Farmer-Friendly Interface** — Built with Streamlit for simplicity and accessibility.

---

## 🛠 Tech Stack
- **Frontend & UI**: Streamlit  
- **AI Framework**: LangChain  
- **LLM Backend**: Groq API (Gemma2-9b-it)  
- **ML Model**: RandomForestClassifier (scikit-learn)  
- **Embeddings**: HuggingFace  
- **Data Handling & Visualization**: Pandas, NumPy, Seaborn  

---

## 📂 Dataset
- Agricultural dataset with features: `Season`, `State`, `Crop`, `Area`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`.
- Target variable: `Yield (tonnes)`.

---

## ⚙️ Installation
```bash
git clone https://github.com/anshKjha10/krishi-mitra-AI.git
cd krishi-mitra-AI
pip install -r requirements.txt

