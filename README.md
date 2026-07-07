# 🚗 Car Price Prediction using Machine Learning

Predict the selling price of used cars using Machine Learning by performing complete data preprocessing, exploratory data analysis (EDA), feature engineering, and regression modeling.

This project compares **Linear Regression** and **Random Forest Regression**, demonstrating how proper preprocessing and target transformation can significantly improve model performance.

---

## 📌 Project Overview

The goal of this project is to predict the selling price of a used car based on various features such as:

- Car Name
- Location
- Manufacturing Year
- Kilometers Driven
- Fuel Type
- Transmission
- Owner Type
- Mileage
- Engine Capacity
- Power
- Number of Seats

The project follows a complete Machine Learning workflow from raw dataset to model evaluation.

---

## 📂 Dataset

Dataset contains information about thousands of used cars including:

- Brand & Model
- Vehicle Age
- Fuel Type
- Transmission
- Owner Type
- Mileage
- Engine
- Power
- Seats
- Selling Price (Target Variable)

---

# 🧠 Machine Learning Pipeline

## 1. Data Cleaning

- Removed unnecessary columns
- Handled missing values
- Checked duplicate records
- Corrected data types

---

## 2. Exploratory Data Analysis (EDA)

Performed visualizations using:

- Histograms
- Countplots
- Boxplots
- Correlation Heatmap
- Scatter Plots
- Pairwise Feature Analysis

Analyzed relationships between features and selling price.

---

## 3. Data Preprocessing

- One-Hot Encoding
- Feature Selection
- Train-Test Split

---

## 4. Model Training

### Linear Regression

Initially trained a basic Linear Regression model.

**Baseline Performance**

- **R² Score:** **57.94%**

---

### Problem Identified

The target variable (**Price**) had a **highly right-skewed distribution**.

Linear Regression is sensitive to skewed targets because it minimizes squared errors, causing expensive luxury cars to dominate the loss.

---

## Solution Implemented

Applied **Log Transformation** on the target variable before training.

```python
model.fit(X_train, np.log1p(y_train))
```

Converted predictions back to original scale using

```python
y_pred = np.expm1(model.predict(X_test))
```

This stabilized variance and reduced the influence of outliers.

---

### Improved Linear Regression Performance

**R² Score:** **80.65%**

✅ Improvement of approximately **23 percentage points**

---

### Random Forest Regression

Trained a Random Forest Regressor for comparison.

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

### Performance

**R² Score:** **89.28%**

Random Forest outperformed Linear Regression by capturing nonlinear relationships within the data.

---

# 📊 Model Comparison

| Model | R² Score |
|--------|---------:|
| Linear Regression (Baseline) | 57.94% |
| Linear Regression (Log Transformed) | 80.65% |
| Random Forest Regression | **89.28%** |

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---


# 📈 Future Improvements

- Hyperparameter tuning using GridSearchCV
- XGBoost Regressor
- CatBoost Regressor
- LightGBM
- Feature Importance Analysis
- Model Deployment using FastAPI
- Interactive Web App using Streamlit

---

# 📚 Key Learnings

- End-to-end Machine Learning workflow
- Data Cleaning & EDA
- Feature Encoding
- Feature Selection
- Regression Models
- Model Evaluation using R² Score
- Importance of target transformation
- Comparison of Linear Regression and Random Forest

---

## ⭐ If you found this project useful, consider giving it a star!
