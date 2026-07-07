import pandas as pd
import numpy as np
import pickle
import json

def load_resources():
    """
    Loads serialized scaler, models, and features info.
    """
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("linear_model.pkl", "rb") as f:
        linear_model = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("features_info.json", "r", encoding="utf-8") as f:
        features_info = json.load(f)
    return scaler, linear_model, rf_model, features_info

def preprocess_input(input_dict, scaler, features_info):
    """
    Preprocesses raw user input into the exact scaled and one-hot encoded vector structure.
    
    input_dict: dict with keys matching ['Brand', 'Location', 'Year', 'Kilometers_Driven', 
                                         'Fuel_Type', 'Transmission', 'Owner_Type', 
                                         'Mileage', 'Engine', 'Power', 'Seats']
    scaler: loaded StandardScaler object
    features_info: loaded features_info dictionary
    """
    # 1. Reconstruct all zero columns
    encoded_columns = features_info["encoded_columns"]
    df_pred = pd.DataFrame(0.0, index=[0], columns=encoded_columns)
    
    # 2. Fill numerical columns from user inputs (or medians if missing)
    numerical_cols = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
    medians = features_info["medians"]
    
    for col in numerical_cols:
        val = input_dict.get(col, None)
        if val is None or pd.isna(val):
            val = medians.get(col, 0.0)
        df_pred.at[0, col] = float(val)
        
    # 3. Apply the fitted StandardScaler on continuous variables
    df_pred[numerical_cols] = scaler.transform(df_pred[numerical_cols])
    
    # 4. Handle categorical features
    # Categories: Brand, Location, Fuel_Type, Transmission, Owner_Type
    cat_cols = ["Brand", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
    for col in cat_cols:
        val = input_dict.get(col, None)
        if val is not None:
            # Construct the dummy variable column name (e.g. "Brand_Audi")
            dummy_col = f"{col}_{val}"
            # Set to 1.0 only if it was not the first class (which is dropped)
            if dummy_col in df_pred.columns:
                df_pred.at[0, dummy_col] = 1.0
                
    return df_pred

def predict_price(input_dict, scaler, linear_model, rf_model, features_info):
    """
    Preprocesses user input and predicts the price using both Linear Regression and Random Forest.
    
    Returns a dict with key predictions.
    """
    df_encoded = preprocess_input(input_dict, scaler, features_info)
    
    # 1. Linear Regression (requires np.expm1 as target was log1p transformed)
    lr_pred_log = linear_model.predict(df_encoded)[0]
    lr_pred = np.expm1(lr_pred_log)
    
    # 2. Random Forest (raw target prediction)
    rf_pred = rf_model.predict(df_encoded)[0]
    
    return {
        "linear_regression": float(lr_pred),
        "random_forest": float(rf_pred)
    }
