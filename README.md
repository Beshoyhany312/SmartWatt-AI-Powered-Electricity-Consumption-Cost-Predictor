# ⚡ SmartWatt — AI-Powered Electricity Consumption & Cost Predictor

SmartWatt is an AI-based system designed to **predict electricity consumption and cost** using Machine Learning and Deep Learning models.  
It helps households and smart-city systems **estimate energy usage, reduce costs, and improve efficiency**.

---

## 🚀 Overview

This project provides an intelligent solution for:

- 🔋 Predicting **electricity consumption (kWh)**
- 💰 Estimating **monthly electricity cost**
- 📊 Analyzing energy usage patterns
- ⚡ Supporting energy-saving decisions

The system is deployed using **Streamlit**, allowing users to interact with the models through a simple web interface.

---

## 🧠 Models Used

### 1. CatBoost Regressor
- Predicts:
  - Electricity Consumption (kWh)
  - Electricity Cost (EGP)
- Works well with structured/tabular data
- Handles categorical features efficiently

### 2. LightGBM Regressor
- Predicts:
  - Electricity Cost (USD)
- Optimized for fast performance and high accuracy

### 3. LSTM (Deep Learning Model)
- Used for:
  - Time-series prediction
  - Learning consumption patterns over time
- Built using TensorFlow/Keras

---

## 📂 Features Used

The models are trained on realistic electricity datasets including:

- Number of appliances (ACs, TVs, refrigerators, etc.)
- Daily usage hours
- House size (m²)
- Seasonal data
- Insulation quality
- Water heater usage
- Washing frequency
- Smart city metrics (for LightGBM model)

---

## 💡 Key Features

- 🔢 Multi-input prediction system
- 📉 12-month consumption forecasting
- 📄 PDF report generation
- 💵 Currency conversion (USD ↔ EGP)
- ⚠️ Budget alert system
- 🌐 Interactive web app (Streamlit)

---

## 🖥️ Demo (Streamlit App)

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
