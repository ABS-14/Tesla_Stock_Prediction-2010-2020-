# Tesla Stock Prediction

This project aims to predict the stock prices of Tesla using historical OHLC (Open, High, Low, Close) data. The dataset used spans from January 1, 2010, to December 31, 2017. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, and applying machine learning models to predict stock price movements.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributors](#contributors)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Tesla_Stock_Prediction.git
   ```
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
   The key dependencies include:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `xgboost`

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Tesla_Stock_Prediction.ipynb
   ```

## Dataset

The dataset used in this project is the historical stock data for Tesla, which includes the following features:
- **Date**
- **Open**: Opening price
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Close**: Closing price
- **Volume**: Number of shares traded

The dataset contains 2,416 rows and 7 columns.

## Exploratory Data Analysis (EDA)

EDA is performed to understand the underlying patterns and distributions in the data. This includes:
- Visualization of the closing prices over time.
- Distribution plots for each feature.
- Box plots to detect outliers.

## Feature Engineering

Feature engineering involves creating new features from the existing ones to enhance the model's predictive power. The following features were created:
- **Day, Month, Year**: Extracted from the date.
- **is_quarter_end**: Indicates if the date is the end of a financial quarter.
- **open-close**: Difference between the opening and closing prices.
- **low-high**: Difference between the lowest and highest prices.
- **target**: Binary target variable indicating whether the next day's closing price is higher than the current day's.

## Model Training

Three machine learning models were trained to predict the stock price movement:
1. **Logistic Regression**
2. **Support Vector Classifier (SVC)**
3. **XGBoost Classifier**

The dataset was split into training and validation sets, with 90% for training and 10% for validation.

## Evaluation

The models were evaluated using ROC-AUC scores and confusion matrices. The XGBoost model showed the best training accuracy, but there was room for improvement in validation accuracy.

## Results

The results of the model training are as follows:
- **Logistic Regression**: 
  - Training Accuracy: 93.8%
  - Validation Accuracy: 44.9%
  
- **SVC**:
  - Training Accuracy: 85.6%
  - Validation Accuracy: 43.2%
  
- **XGBoost Classifier**:
  - Training Accuracy: 94.9%
  - Validation Accuracy: 45.9%

## Contributors

- Your Name - [GitHub](https://github.com/yourusername)
  
Feel free to contribute to this project by submitting issues or pull requests.

---
