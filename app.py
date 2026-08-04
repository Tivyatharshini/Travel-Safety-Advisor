"""
Travel Safety Advisor - cleaned up single app.

Fixes vs. the original app files:
  - final.py only collected 4 of the 27 features the model needed and
    silently zero-filled the rest (state, district, case counts, date,
    coordinates). That's why predictions were unreliable -- this app
    now collects every feature the model actually uses.
  - app.py was an incomplete stub with a hardcoded Windows path - removed.
  - travel.py crashed on unseen states/districts (LabelEncoder.transform
    raises on unseen labels) - now handled via a dropdown of real values
    plus a graceful "unknown" fallback if you type a new one.
  - All paths are relative, so this runs on any machine/OS.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

BASE = Path(__file__).parent
SAVED = BASE / "saved"

model = joblib.load(SAVED / "random_forest_model.joblib")
scaler = joblib.load(SAVED / "scaler.joblib")
imputer = joblib.load(SAVED / "imputer.joblib")
feature_names = joblib.load(SAVED / "feature_names.joblib")
encoders = joblib.load(SAVED / "encoders.joblib")
state_options = joblib.load(SAVED / "state_options.joblib")
district_by_state = joblib.load(SAVED / "district_by_state.joblib")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

HEALTH_FLAGS = [f for f in feature_names if f not in (
    "week_of_outbreak", "state_ut", "district", "Cases", "day", "mon",
    "year", "Latitude", "Longitude", "preci", "LAI", "Temp", "AQI",
)]


def encode_categorical(col, value):
    """Map to the encoder's classes; fall back to '__UNKNOWN__' for new values."""
    le = encoders[col]
    value = str(value)
    if value not in le.classes_:
        value = "__UNKNOWN__"
    return le.transform([value])[0]


def get_weather(city):
    if not OPENWEATHER_API_KEY:
        return None
    try:
        r = requests.get(
            WEATHER_API_URL,
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def main():
    st.set_page_config(page_title="Travel Safety Advisor", page_icon="\U0001f30d", layout="wide")
    st.title("\U0001f30d Travel Safety Advisor")
    st.caption("Predicts travel safety from location, timing, environment, and your health profile.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trip details")
        state = st.selectbox("State/UT", state_options)
        districts = district_by_state.get(state, [])
        district = st.selectbox("District", districts) if districts else st.text_input("District")
        date = st.date_input("Travel date", datetime.today())

        city_for_weather = st.text_input("City/Town (for live weather lookup, optional)", district)
        weather = get_weather(city_for_weather) if city_for_weather else None

        if weather and "main" in weather:
            st.success("Live weather retrieved.")
            temp_c = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            st.write(f"\U0001f321\ufe0f {temp_c:.1f}\u00b0C \u00b7 \U0001f4a7 {humidity}% humidity")
            temp_kelvin = temp_c + 273.15
        else:
            st.info("No live weather (missing API key, offline, or city not found) - enter manually.")
            temp_c = st.slider("Temperature (\u00b0C)", -10.0, 50.0, 28.0)
            temp_kelvin = temp_c + 273.15

        aqi = st.slider("Air Quality Index (0-500)", 0, 500, 100)
        preci = st.slider("Precipitation (mm)", 0.0, 500.0, 10.0)
        lai = st.slider("Leaf Area Index (0-10)", 0.0, 10.0, 2.5)

        st.subheader("Outbreak context")
        cases = st.number_input("Recently reported cases in the region (0 if unknown)", min_value=0, value=0, step=1)
        latitude = st.number_input("Latitude", value=20.0, format="%.4f")
        longitude = st.number_input("Longitude", value=78.0, format="%.4f")

    with col2:
        st.subheader("Your health conditions")
        st.caption("Check anything that applies to you.")
        input_flags = {}
        cols = st.columns(2)
        for i, flag in enumerate(HEALTH_FLAGS):
            with cols[i % 2]:
                input_flags[flag] = 1 if st.checkbox(flag.replace("_", " ").title()) else 0

    st.divider()

    if st.button("Predict Safety", type="primary"):
        week_of_outbreak = f"{date.isocalendar().week}th week"  # matches training format loosely; unknown -> fallback bucket

        row = {
            "week_of_outbreak": encode_categorical("week_of_outbreak", week_of_outbreak),
            "state_ut": encode_categorical("state_ut", state),
            "district": encode_categorical("district", district),
            "Cases": cases,
            "day": date.day,
            "mon": date.month,
            "year": date.year,
            "Latitude": latitude,
            "Longitude": longitude,
            "preci": preci,
            "LAI": lai,
            "Temp": temp_kelvin,
            "AQI": aqi,
        }
        row.update(input_flags)

        input_df = pd.DataFrame([row])[feature_names]
        X_imp = imputer.transform(input_df)
        X_scaled = scaler.transform(X_imp)

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        safe_prob = proba[1] * 100

        st.subheader("Result")
        if pred == 1:
            st.success(f"\u2705 {district or state} looks SAFE for travel under these conditions.")
        else:
            st.error(f"\u26a0\ufe0f {district or state} might NOT be safe for travel under these conditions.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=safe_prob,
            title={"text": "Safety Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkgreen"},
                "steps": [
                    {"range": [0, 33], "color": "red"},
                    {"range": [33, 66], "color": "yellow"},
                    {"range": [66, 100], "color": "green"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Why? (top model factors)"):
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            st.bar_chart(importances.head(8))


if __name__ == "__main__":
    main()
