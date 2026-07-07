import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_resources, predict_price

# Set page config for a wide premium dashboard layout, with sidebar collapsed
st.set_page_config(
    page_title="ValuCar - AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide sidebar and toggle button completely */
    [data-testid="collapsedControl"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #3DDEFF 0%, #4ACCED 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        text-align: center;
    }
    .header-container h1 {
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .header-container p {
        font-weight: 300;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card design */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #2a5298;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    .rf-card {
        border-left: 5px solid #00b4db;
    }
    .rf-card p {
        color: #00b4db;
    }
    
    /* Info box styling */
    .info-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        margin-top: 1.5rem;
    }
    
    /* Input Section Header styling */
    .section-header {
        font-weight: 700;
        font-size: 1.4rem;
        color: #4D86EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4D86EB 0%, #3DDEFF 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(42, 82, 152, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="header-container">
    <h1>🚗 ValuCar AI</h1>
    <p>Predict Used Car Market Values in India instantly using Machine Learning models</p>
</div>
""", unsafe_allow_html=True)

# Load machine learning assets
@st.cache_resource
def load_all_assets():
    try:
        return load_resources()
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None

scaler, linear_model, rf_model, features_info = load_all_assets()

if scaler is not None:
    categorical_options = features_info["categorical_options"]
    medians = features_info["medians"]
    
    # Create Tabs for Prediction and Model Stats
    tab_predict, tab_stats = st.tabs(["🔮 Predict Car Value", "📊 Model Performance Stats"])
    
    with tab_predict:
        st.markdown('<div class="section-header">1. Configure Car Specifications</div>', unsafe_allow_html=True)
        
        # Row 1: Brand, Year, Location
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            brand = st.selectbox("Brand", options=categorical_options["Brand"], index=categorical_options["Brand"].index("Hyundai") if "Hyundai" in categorical_options["Brand"] else 0)
        with row1_col2:
            year = st.slider("Manufacturing Year", min_value=1998, max_value=2020, value=2014, step=1)
        with row1_col3:
            location = st.selectbox("Location", options=categorical_options["Location"], index=categorical_options["Location"].index("Mumbai") if "Mumbai" in categorical_options["Location"] else 0)
            
        # Row 2: Fuel Type, Transmission, Owner Type
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            fuel_type = st.selectbox("Fuel Type", options=categorical_options["Fuel_Type"], index=categorical_options["Fuel_Type"].index("Diesel") if "Diesel" in categorical_options["Fuel_Type"] else 0)
        with row2_col2:
            transmission = st.selectbox("Transmission", options=categorical_options["Transmission"], index=categorical_options["Transmission"].index("Manual") if "Manual" in categorical_options["Transmission"] else 0)
        with row2_col3:
            owner_type = st.selectbox("Owner Type", options=categorical_options["Owner_Type"], index=categorical_options["Owner_Type"].index("First") if "First" in categorical_options["Owner_Type"] else 0)
            
        # Row 3: Mileage, Engine Capacity, Power
        row3_col1, row3_col2, row3_col3 = st.columns(3)
        with row3_col1:
            mileage = st.slider("Mileage (kmpl or km/kg)", min_value=5.0, max_value=40.0, value=18.0, step=0.1)
        with row3_col2:
            engine = st.slider("Engine Capacity (CC)", min_value=500, max_value=6000, value=1500, step=50)
        with row3_col3:
            power = st.slider("Power (bhp)", min_value=30.0, max_value=600.0, value=100.0, step=1.0)
            
        # Row 4: Kilometers Driven, Seats
        row4_col1, row4_col2 = st.columns(2)
        with row4_col1:
            km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=1000000, value=50000, step=1000)
        with row4_col2:
            seats = st.slider("Seats", min_value=2, max_value=10, value=5, step=1)
            
        # Store inputs in a dict
        user_inputs = {
            "Brand": brand,
            "Location": location,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Owner_Type": owner_type,
            "Year": year,
            "Kilometers_Driven": km_driven,
            "Mileage": mileage,
            "Engine": engine,
            "Power": power,
            "Seats": seats
        }
        
        st.markdown('<div class="section-header">2. Estimated Valuation</div>', unsafe_allow_html=True)
        
        # Make predictions
        predictions = predict_price(user_inputs, scaler, linear_model, rf_model, features_info)
        
        # Display predictions side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Linear Regression Prediction</h3>
                <p>₹ {predictions['linear_regression']:.2f} Lakh</p>
                <small style="color: #666;">R² Accuracy: {features_info['metrics']['linear_regression']['r2']*100:.2f}%</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card rf-card">
                <h3>Random Forest Prediction</h3>
                <p>₹ {predictions['random_forest']:.2f} Lakh</p>
                <small style="color: #666;">R² Accuracy: {features_info['metrics']['random_forest']['r2']*100:.2f}%</small>
            </div>
            """, unsafe_allow_html=True)
            
        # Summary and recommendation box
        st.markdown('<div class="info-container">', unsafe_allow_html=True)
        st.markdown("##### 💡 Recommendation Guide")
        diff = abs(predictions['random_forest'] - predictions['linear_regression'])
        avg_price = (predictions['random_forest'] + predictions['linear_regression']) / 2
        st.write(
            f"The **Random Forest** model is generally more robust and accurate (R²: ~89.2%). "
            f"The estimated average valuation of your car is **₹ {avg_price:.2f} Lakhs**. "
            f"There is a difference of **₹ {diff:.2f} Lakhs** between the two predictions. "
            "For premium high-power cars, the Random Forest model captures nonlinear pricing structures better, "
            "while the Linear Regression model offers a smoother, log-normalized price expectation."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display entered parameters for user review
        st.markdown("##### Car Configuration Summary")
        summary_df = pd.DataFrame([user_inputs])
        st.dataframe(summary_df, use_container_width=True)
        
    with tab_stats:
        st.markdown("### Model Validation Performance Metrics")
        st.write("Below are the metrics computed on the test split (20% of the dataset).")
        
        metrics = features_info["metrics"]
        
        # Convert metrics to a nice dataframe
        metrics_df = pd.DataFrame({
            "Metric": ["R² Score (Accuracy)", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
            "Linear Regression (Baseline)": [
                f"{metrics['linear_regression']['r2']*100:.2f}%",
                f"₹ {metrics['linear_regression']['mae']:.3f} Lakhs",
                f"₹ {metrics['linear_regression']['rmse']:.3f} Lakhs"
            ],
            "Random Forest Regressor": [
                f"{metrics['random_forest']['r2']*100:.2f}%",
                f"₹ {metrics['random_forest']['mae']:.3f} Lakhs",
                f"₹ {metrics['random_forest']['rmse']:.3f} Lakhs"
            ]
        })
        
        st.table(metrics_df)
        
        # Plot R2 comparison
        st.markdown("##### Visual Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 3))
        models_list = ["Linear Regression", "Random Forest"]
        r2_scores = [metrics['linear_regression']['r2'] * 100, metrics['random_forest']['r2'] * 100]
        sns.barplot(x=models_list, y=r2_scores, palette="Blues_d", ax=ax)
        ax.set_ylabel("R² Score (%)")
        ax.set_ylim(0, 100)
        for idx, val in enumerate(r2_scores):
            ax.text(idx, val + 2, f"{val:.2f}%", ha='center', fontweight='bold')
        st.pyplot(fig)
else:
    st.warning("Please ensure train_and_save_models.py has been run successfully to create serialize models and scalers.")
