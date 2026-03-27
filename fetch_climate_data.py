import os
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def fetch_realtime_climate_data(locations, filename="spatiotemporal_climate_data.csv"):
    print("🔄 INITIALIZING REAL-TIME DATA PIPELINE...")

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        print(f"Loaded existing cache with {len(existing_df)} records.")
    else:
        existing_df = pd.DataFrame()
        print("No cache found. Preparing for deep historical fetch (10 years).")

    all_new_data = []
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    for loc in locations:
        lat, lon, city = loc["lat"], loc["lon"], loc["name"]
        
        # --- PART A: THE ARCHIVE FETCH ---
        if existing_df.empty or city not in existing_df["City"].values:
            start_date_archive = (today - timedelta(days=3650)).strftime("%Y-%m-%d")
            end_date_archive = (today - timedelta(days=6)).strftime("%Y-%m-%d")
            
            print(f"[{city}] 📚 Fetching deep history...")
            
            archive_url = "https://archive-api.open-meteo.com/v1/archive"
            archive_params = {
                "latitude": lat, "longitude": lon,
                "start_date": start_date_archive, "end_date": end_date_archive,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
                "timezone": "auto",
            }
            
            try:
                time.sleep(1)
                resp = session.get(archive_url, params=archive_params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    df_archive = pd.DataFrame({
                        "Date": pd.to_datetime(data["daily"]["time"]),
                        "City": city, "Latitude": lat, "Longitude": lon,
                        "Max_Temp_C": data["daily"]["temperature_2m_max"],
                        "Min_Temp_C": data["daily"]["temperature_2m_min"],
                        "Precipitation_mm": data["daily"]["precipitation_sum"],
                        "Wind_Speed": data["daily"]["wind_speed_10m_max"]
                    })
                    all_new_data.append(df_archive)
            except Exception as e:
                print(f"[{city}] Archive Error: {e}")

        # --- PART B: THE LIVE FETCH ---
        # Expanded to 30 days to guarantee the 21-day rolling math works flawlessly
        past_days = 30
        print(f"[{city}] ⚡ Fetching LIVE data (Last {past_days} days)")
        
        live_url = "https://api.open-meteo.com/v1/forecast"
        live_params = {
            "latitude": lat, "longitude": lon,
            "past_days": past_days, 
            "forecast_days": 1,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
            "timezone": "auto",
        }
        
        try:
            time.sleep(1)
            resp = session.get(live_url, params=live_params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                df_live = pd.DataFrame({
                    "Date": pd.to_datetime(data["daily"]["time"]),
                    "City": city, "Latitude": lat, "Longitude": lon,
                    "Max_Temp_C": data["daily"]["temperature_2m_max"],
                    "Min_Temp_C": data["daily"]["temperature_2m_min"],
                    "Precipitation_mm": data["daily"]["precipitation_sum"],
                    "Wind_Speed": data["daily"]["wind_speed_10m_max"]
                })
                all_new_data.append(df_live)
        except Exception as e:
            print(f"[{city}] Live Fetch Error: {e}")

    # --- PART C: MERGE AND CLEANUP ---
    if all_new_data:
        new_df = pd.concat(all_new_data, ignore_index=True)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["Date", "City"], keep="last", inplace=True)
        combined_df.sort_values(by=["City", "Date"], inplace=True)
        combined_df.to_csv(filename, index=False)
        print(f"\n✅ SUCCESS! Database updated. Total records: {len(combined_df)}")
        return combined_df
    else:
        print("\n⚠️ No data fetched. Check connection.")
        return existing_df

locations_to_monitor = [
    {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
]

if __name__ == "__main__":
    fetch_realtime_climate_data(locations_to_monitor)