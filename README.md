# 🚀 Crypto AI Trading Bot (BTC Prediction System)

An advanced, production-grade Bitcoin prediction system built using Machine Learning and Quantitative Trading techniques.
This project focuses on generating high-confidence trading signals using a hybrid ensemble of models and robust data engineering.

---

## 📌 Overview

This system predicts high-probability market movements in Bitcoin (BTC/USDT) using:

* 2 years of historical market data
* Advanced feature engineering (technical + market data)
* Triple Barrier labeling strategy
* Walk-forward validation (time-series aware)
* Ensemble learning with model stacking

The goal is not just prediction, but **high-quality signal generation with controlled risk**.

---

## 🧠 Implemented Model Architecture

### 🔹 Core Models

* **XGBoost** → Handles structured/tabular features efficiently
* **LightGBM** → Fast gradient boosting with improved generalization

### 🔹 Ensemble Strategy (Stacking)

Instead of simple averaging, we use a **Meta Model**:

* Base Models:

  * XGBoost
  * LightGBM
* Meta Model:

  * Logistic Regression

This allows the system to learn **how to combine predictions intelligently**, improving accuracy and reducing noise.

---

## ⚙️ Key Techniques Used

### 📊 1. Data Engineering

* 2 years of hourly OHLCV data from Binance API
* Volume delta & order flow features
* Funding rate integration
* Fear & Greed Index sentiment

---

### 🧪 2. Feature Engineering

* RSI, MACD, EMA (21, 50, 200)
* Bollinger Bands (width, squeeze)
* ATR (volatility)
* VWAP-based features
* Volume & regime detection

---

### 🎯 3. Labeling Strategy

* **Triple Barrier Method**

  * Take Profit (TP): 1.5 × ATR
  * Stop Loss (SL): 1 × ATR
  * Time Horizon: 6 hours

This provides realistic trade outcomes instead of naive up/down labels.

---

### 🔁 4. Walk-Forward Validation

* TimeSeriesSplit (5 folds)
* Prevents data leakage
* Simulates real-world trading conditions

---

### 🎛️ 5. Probability Calibration

* Isotonic Regression
* Produces reliable confidence scores

---

### 🎚️ 6. Threshold Optimization

* Precision-Recall based threshold selection
* Focus on high-quality signals instead of quantity

---

## 📈 Model Performance

| Metric      | Value |
| ----------- | ----- |
| CV AUC      | ~0.55 |
| Test AUC    | ~0.54 |
| Brier Score | ~0.18 |

> Note: Financial markets are highly noisy. Even small predictive edges can be profitable when combined with proper risk management.

---

## 🧠 Trading Logic

* Model outputs probability of upward movement
* Signals are filtered based on:

  * Confidence threshold
  * Risk conditions
* Designed to generate **high-confidence trades only**

---

## 📂 Project Structure

```bash
btc_v2_final/
│
├── models/
│   └── trainer.py        # Model training pipeline
│
├── features/
│   └── engineer.py       # Feature engineering
│
├── retrain/
│   └── scheduler.py      # Auto retraining logic
│
├── risk/
│   └── manager.py        # Risk management
│
├── storage/
│   └── db.py             # Database handling
│
├── config/
│   └── settings.py       # Configurations
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python models/trainer.py
```

---

## ⚠️ Important Notes

* This project is for **educational and research purposes**
* Not financial advice
* Real trading involves risk

---

## 🔮 Future Improvements

* Add LSTM / Transformer models
* Reinforcement Learning trading agent
* Live trading integration (Binance API)
* Dashboard (React + FastAPI)
* Multi-timeframe analysis

---

## 👨‍💻 Author

**Shubham Yadav**
B.Tech IT | AI & ML Enthusiast

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub and support the work!
