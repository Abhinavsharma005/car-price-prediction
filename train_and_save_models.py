import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# 2. Drop columns
print("Dropping unnecessary columns...")
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)
if "New_Price" in df.columns:
    df.drop("New_Price", axis=1, inplace=True)

# 3. Clean units
print("Cleaning units and converting to numeric...")
df["Mileage"] = df["Mileage"].str.replace(" kmpl", "", regex=False)
df["Mileage"] = df["Mileage"].str.replace(" km/kg", "", regex=False)
df["Engine"] = df["Engine"].str.replace(" CC", "", regex=False)
df["Power"] = df["Power"].str.replace(" bhp", "", regex=False)

df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Engine"] = pd.to_numeric(df["Engine"], errors="coerce")
df["Power"] = pd.to_numeric(df["Power"], errors="coerce")

# 4. Handle invalid zeros as NaN
df.loc[df["Mileage"] == 0, "Mileage"] = np.nan
df.loc[df["Seats"] == 0, "Seats"] = np.nan

# 5. Compute medians from the dataset before filling (to save for streamlit app default/imputation fallback)
medians = {
    "Mileage": float(df["Mileage"].median()),
    "Engine": float(df["Engine"].median()),
    "Power": float(df["Power"].median()),
    "Seats": float(df["Seats"].median()),
    "Year": float(df["Year"].median()),
    "Kilometers_Driven": float(df["Kilometers_Driven"].median())
}
print(f"Computed medians: {medians}")

# Fill missing values
df["Mileage"] = df["Mileage"].fillna(medians["Mileage"])
df["Engine"] = df["Engine"].fillna(medians["Engine"])
df["Power"] = df["Power"].fillna(medians["Power"])
df["Seats"] = df["Seats"].fillna(medians["Seats"])

# 6. Extract Brand and drop Name
print("Extracting Brand and dropping Name...")
df["Brand"] = df["Name"].str.split().str[0]
df.drop("Name", axis=1, inplace=True)

# Save unique categories for Streamlit selectbox options
cat_cols = ["Brand", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
categorical_options = {}
for col in cat_cols:
    categorical_options[col] = sorted(df[col].dropna().unique().tolist())

# 7. One-hot encode categoricals
print("Encoding categorical variables...")
x = df.drop(columns=["Price"], axis=1)
y = df["Price"]
X_one_encode = pd.get_dummies(x, columns=cat_cols, drop_first=True)

# 8. Scale numerical features
print("Scaling numerical columns...")
numerical_cols = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
scaler = StandardScaler()
X_one_encode[numerical_cols] = scaler.fit_transform(X_one_encode[numerical_cols])

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler.pkl")

# 9. Train/Test Split
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_one_encode, y, test_size=0.2, random_state=42)

# 10. Fit models
print("Training Linear Regression model on log-transformed target...")
linear_model = LinearRegression()
linear_model.fit(X_train, np.log1p(y_train))

# Evaluate Linear Regression
lr_pred_log = linear_model.predict(X_test)
lr_pred = np.expm1(lr_pred_log)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = root_mean_squared_error(y_test, lr_pred)
print(f"Linear Regression R2: {lr_r2:.4f}, MAE: {lr_mae:.4f}, RMSE: {lr_rmse:.4f}")

print("Training Random Forest model on raw target...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = root_mean_squared_error(y_test, rf_pred)
print(f"Random Forest R2: {rf_r2:.4f}, MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}")

# Save models
with open("linear_model.pkl", "wb") as f:
    pickle.dump(linear_model, f)
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Saved linear_model.pkl and rf_model.pkl")

# Save features info and metadata
features_info = {
    "encoded_columns": X_one_encode.columns.tolist(),
    "categorical_options": categorical_options,
    "medians": medians,
    "metrics": {
        "linear_regression": {
            "r2": float(lr_r2),
            "mae": float(lr_mae),
            "rmse": float(lr_rmse)
        },
        "random_forest": {
            "r2": float(rf_r2),
            "mae": float(rf_mae),
            "rmse": float(rf_rmse)
        }
    }
}

with open("features_info.json", "w", encoding="utf-8") as f:
    json.dump(features_info, f, indent=4)
print("Saved features_info.json")
print("Training and serialization completed successfully!")
