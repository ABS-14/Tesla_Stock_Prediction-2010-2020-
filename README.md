# Tesla Stock Directional Forecaster 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Research_Prototype-yellow)

A machine learning system designed to predict the **directional movement** (Up/Down) of Tesla (TSLA) stock prices. Unlike standard regression models that attempt to predict exact prices (which is often noisy), this project treats the problem as a **Binary Classification** task, leveraging **XGBoost** and engineered technical indicators to identify profitable trading signals.

## Project Overview

Financial time-series data is inherently non-stationary and noisy. This project analyzes **8 years of historical OHLC data** (2,416 trading days) to engineer features that capture market psychology—specifically volatility and momentum. 

The core objective is to validate whether machine learning classifiers can outperform random-walk baselines in predicting short-term price trends.

### Key Features
* **Time-Series Feature Engineering:** distinct focus on derived metrics (Volatility, Momentum) rather than raw prices to reduce non-stationarity.
* **Ensemble Learning:** Implements and compares **XGBoost**, **Support Vector Classifiers (SVC)**, and **Logistic Regression**.
* **Rigorous Validation:** Adheres to strict **chronological data splitting** to prevent "Look-Ahead Bias" (Data Leakage), ensuring the model is evaluated only on unseen future data.
* **Calendar Anomalies:** Incorporates institutional trading patterns via "Quarter-End" signaling.

---

## Dataset & Engineering

The system processes daily trading data for Tesla (TSLA) from **2010 to 2017**.

**Raw Input:**
* Open, High, Low, Close, Volume

**Engineered Technical Indicators:**
Instead of feeding raw prices to the model, we calculate derivatives that represent market state:
1.  **Price Momentum (`Open - Close`):** Captures the daily directional strength. A large positive value indicates selling pressure; a negative value indicates buying pressure.
2.  **Intraday Volatility (`High - Low`):** Measures the daily trading range. High values signal market uncertainty or news events.
3.  **Seasonality (`is_quarter_end`):** A binary flag for the last month of a financial quarter. This captures "Window Dressing" effects where institutional investors rebalance portfolios, often leading to predictable price anomalies.

---

## Technical Architecture

### 1. Preprocessing Pipeline
* **Stationarity Transformation:** Converted absolute prices into relative differences (deltas) to make the dataset suitable for ML algorithms.
* **Feature Scaling:** Applied `StandardScaler` to normalize volatility and momentum features, ensuring outliers (e.g., earnings calls) do not skew the gradient descent process.

### 2. Model Selection
We trained and evaluated three distinct architectures to compare linear vs. non-linear capabilities:
* **Logistic Regression:** Baseline linear model.
* **SVC (Polynomial Kernel):** Captures non-linear decision boundaries.
* **XGBoost (Extreme Gradient Boosting):** Tree-based ensemble method chosen for its ability to handle feature interactions and resistance to overfitting.

### 3. Evaluation Protocol
* **Metric:** **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve) was used instead of simple accuracy, as it provides a more robust measure of the model's ability to distinguish between "Up" and "Down" days.
* **Validation Strategy:** Utilized a strict **Time-Series Split** (Train: 2010-2016, Test: 2017). *Note: Random shuffling was explicitly disabled to prevent data leakage.*

---

## Installation & Usage

### Prerequisites
* Python 3.x
* XGBoost, Scikit-Learn, Pandas, Matplotlib

### Running the Forecaster
```python
import pandas as pd
from xgboost import XGBClassifier

# 1. Load Data
df = pd.read_csv('TSLA.csv')

# 2. Preprocess & Engineer Features
df['momentum'] = df['Open'] - df['Close']
df['volatility'] = df['High'] - df['Low']
# ... (rest of engineering pipeline)

# 3. Train Model
model = XGBClassifier()
model.fit(X_train, Y_train)

# 4. Predict Direction (1 = Up, 0 = Down)
prediction = model.predict(X_test)
