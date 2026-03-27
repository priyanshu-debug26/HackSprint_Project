import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn
from datetime import datetime

warnings.filterwarnings('ignore')

GLOBAL_TEMP_MODEL = None
GLOBAL_RAIN_MODEL = None
FEATURES = []

def train_emergency_brain(input_file="spatiotemporal_climate_data.csv"):
    global GLOBAL_TEMP_MODEL, GLOBAL_RAIN_MODEL, FEATURES
    print("🚨 INITIALIZING EMERGENCY AI CORE 🚨")

    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['City', 'Date']).reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Run fetch_climate_data.py first.")
        return

    print("Analyzing 21-day micro-seasonal baselines...")
    
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.0)
    df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.0)
    
    df['Temp_Yesterday'] = df.groupby('City')['Max_Temp_C'].shift(1)
    df['Rain_Yesterday'] = df.groupby('City')['Precipitation_mm'].shift(1)
    
    # UPGRADE: 21-Day Rolling Baselines
    df['Rolling_21_Day_Temp'] = df.groupby('City')['Max_Temp_C'].transform(lambda x: x.rolling(window=21).mean())
    df['Rolling_21_Day_Rain'] = df.groupby('City')['Precipitation_mm'].transform(lambda x: x.rolling(window=21).sum())
    
    df['Target_Next_Day_Temp'] = df.groupby('City')['Max_Temp_C'].shift(-1)
    df['Target_Next_Day_Rain'] = df.groupby('City')['Precipitation_mm'].shift(-1)

    model_df = df.dropna().copy()

    FEATURES = [
        'Day_Sin', 'Day_Cos', 'Max_Temp_C', 'Min_Temp_C', 'Precipitation_mm', 
        'Wind_Speed', 'Temp_Yesterday', 'Rain_Yesterday', 'Rolling_21_Day_Temp', 'Rolling_21_Day_Rain'
    ]
    
    X = model_df[FEATURES]
    y_temp = model_df['Target_Next_Day_Temp']
    y_rain = model_df['Target_Next_Day_Rain']

    print("\nIgniting Cores... Training Dual-Threat Models")
    
    temp_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, tree_method='hist', random_state=42)
    temp_model.fit(X, y_temp, verbose=False)
    
    rain_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, tree_method='hist', random_state=42)
    rain_model.fit(X, y_rain, verbose=False)

    GLOBAL_TEMP_MODEL = temp_model
    GLOBAL_RAIN_MODEL = rain_model
    print("✅ Models Trained and Loaded into Memory.")

# --- FASTAPI SERVER SETUP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

@app.get("/predict")
def predict_threat(lat: float, lon: float):
    if GLOBAL_TEMP_MODEL is None:
        return {"error": "Model not initialized."}

    # Fetch 25 days so we can cleanly calculate the 21-day rolling average for 'today'
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=25&forecast_days=1&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=auto"
    try:
        res = requests.get(url).json()
        daily = res['daily']
        today_idx = len(daily['time']) - 1
        
        day_of_year = datetime.now().timetuple().tm_yday
        max_temp = daily['temperature_2m_max'][today_idx]
        
        # Calculate 21-day rolling baseline
        rolling_temp_21 = sum(daily['temperature_2m_max'][today_idx-20:today_idx+1]) / 21
        rolling_rain_21 = sum(daily['precipitation_sum'][today_idx-20:today_idx+1])

        live_features = pd.DataFrame([{
            'Day_Sin': np.sin(2 * np.pi * day_of_year / 365.0),
            'Day_Cos': np.cos(2 * np.pi * day_of_year / 365.0),
            'Max_Temp_C': max_temp,
            'Min_Temp_C': daily['temperature_2m_min'][today_idx],
            'Precipitation_mm': daily['precipitation_sum'][today_idx],
            'Wind_Speed': daily['wind_speed_10m_max'][today_idx],
            'Temp_Yesterday': daily['temperature_2m_max'][today_idx - 1],
            'Rain_Yesterday': daily['precipitation_sum'][today_idx - 1],
            'Rolling_21_Day_Temp': rolling_temp_21,
            'Rolling_21_Day_Rain': rolling_rain_21
        }])

        pred_temp = float(GLOBAL_TEMP_MODEL.predict(live_features)[0])
        pred_rain = float(GLOBAL_RAIN_MODEL.predict(live_features)[0])

        status = "🟢 STABLE CLIMATE"
        hex_color = "#4ade80"
        threat_level = 0
        icon = "🟢"

        # Math Update: 3.5C over a 21-day average is highly anomalous
        if pred_rain > 50.0:
            status = "🌊 CRITICAL FLOOD RISK"
            hex_color = "#00bfff"
            threat_level = 5 
            icon = "🌊"
        elif pred_temp >= rolling_temp_21 + 3.5:
            status = "🔥 SUDDEN HEAT SPIKE"
            hex_color = "#ef4444"
            threat_level = 4
            icon = "🔥"
        elif pred_temp > 40.0:
            status = "🔥 SEVERE HEATWAVE"
            hex_color = "#ff8a00"
            threat_level = 3
            icon = "⚠️"

        return {
            "predicted_temp": round(pred_temp, 1),
            "predicted_rain": round(pred_rain, 1),
            "current_temp": max_temp,
            "baseline_21_temp": round(rolling_temp_21, 1),
            "status": status,
            "hex_color": hex_color,
            "threat_level": threat_level,
            "icon": icon,
            "times": daily['time'],
            "temps": daily['temperature_2m_max'],
            "mins": daily['temperature_2m_min'],
            "rains": daily['precipitation_sum']
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    train_emergency_brain()
    print("📡 Starting API Link on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)