import ee
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import datetime
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json
import os
import tempfile

# -----------------------------------------------------------------------------
# Step A: Compute milliseconds until next midnight (12 AM)
# -----------------------------------------------------------------------------
now = datetime.datetime.now()
tomorrow = now + datetime.timedelta(days=1)
midnight = datetime.datetime.combine(tomorrow.date(), datetime.time.min)
ms_until_midnight = int((midnight - now).total_seconds() * 1000)

# Auto-refresh the app at next midnight.
st_autorefresh(interval=ms_until_midnight, limit=1, key="autoRefresh")

# -----------------------------------------------------------------------------
# Step 1: Initialize Earth Engine with your service account credentials
# -----------------------------------------------------------------------------
if "ee_credentials" not in st.secrets:
    st.error("Missing 'ee_credentials' in secrets. Please add your service account JSON as a single string under the 'json' key.")
    st.stop()

# Retrieve the key data from Streamlit secrets.
key_data = st.secrets["ee_credentials"]["json"]

# Ensure key_data is a string.
if isinstance(key_data, dict):
    key_data = json.dumps(key_data)
else:
    key_data = key_data.strip()

# Parse the JSON to extract the service account email.
try:
    creds_dict = json.loads(key_data)
except Exception as e:
    st.error(f"Error parsing key_data: {e}")
    st.stop()

service_account_email = creds_dict["client_email"]

# Write the JSON key to a temporary file.
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
    tmp_file.write(key_data)
    tmp_file_path = tmp_file.name

st.write("Using temporary key file:", tmp_file_path)

# Attempt to authenticate using the temporary file.
try:
    credentials = ee.ServiceAccountCredentials(service_account_email, tmp_file_path)
    ee.Initialize(credentials)
    st.success("✅ Google Earth Engine authenticated successfully!")
except Exception as e:
    st.error(f"❌ Earth Engine authentication failed: {e}")
    st.info("Please ensure your service account JSON is exactly as provided by Google Cloud Console.")
    st.stop()
finally:
    # Optionally remove the temporary file.
    try:
        os.remove(tmp_file_path)
    except Exception as e:
        st.warning(f"Could not remove temporary key file: {e}")

# -----------------------------------------------------------------------------
# Step 2: Define Parameters and Region of Interest (ROI)
# -----------------------------------------------------------------------------
region = ee.Geometry.Rectangle([76.3, 18.2, 76.7, 18.6])
today = datetime.date.today()
twelve_months_ago = today - relativedelta(months=12)
start_date = twelve_months_ago.strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")
st.write(f"**Data Range for Model:** {start_date} to {end_date}")

# -----------------------------------------------------------------------------
# Step 3: Define Function to Retrieve, Process, and Forecast Data with LSTM
# -----------------------------------------------------------------------------
@st.cache_data(ttl=ms_until_midnight/1000, show_spinner=True)
def get_forecast():
    smap_collection = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007").filterDate(start_date, end_date)
    
    def compute_sm_feature(image):
        mean_surface = image.select("sm_surface").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            bestEffort=True,
            maxPixels=1e9
        ).get("sm_surface")
        mean_rootzone = image.select("sm_rootzone").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            bestEffort=True,
            maxPixels=1e9
        ).get("sm_rootzone")
        isInvalid = ee.Algorithms.If(
            ee.Algorithms.IsEqual(mean_surface, None),
            True,
            ee.Algorithms.IsEqual(mean_rootzone, None)
        )
        combined_sm = ee.Algorithms.If(
            isInvalid,
            None,
            ee.Number(mean_surface).add(ee.Number(mean_rootzone)).divide(2)
        )
        date_str = image.date().format("YYYY-MM-dd")
        return ee.Feature(None, {"date": date_str, "sm": combined_sm})
    
    sm_features = smap_collection.map(compute_sm_feature)
    sm_features = sm_features.filter(ee.Filter.notNull(["sm"]))
    
    dates_list = sm_features.aggregate_array("date").getInfo()
    sm_list = sm_features.aggregate_array("sm").getInfo()
    
    df = pd.DataFrame({"ds": dates_list, "y": sm_list})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    
    window_size = 7
    def create_sequences(data, dates, window_size):
        X, y, target_dates = [], [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
            target_dates.append(dates.iloc[i+window_size])
        return np.array(X), np.array(y), pd.Series(target_dates)
    
    X_all, y_all, target_dates = create_sequences(df["y"].values, df["ds"], window_size)
    target_dates = pd.to_datetime(target_dates)
    
    X_train = X_all.reshape(-1, window_size, 1)
    y_train = y_all
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
    lstm_model.add(LSTM(25, activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    
    y_train_pred = lstm_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    st.write("**Training RMSE (LSTM):**", train_rmse)
    
    last_window = df["y"].values[-window_size:]
    extended = list(last_window)
    last_date_forecast = df["ds"].iloc[-1]
    future_preds = []
    
    for i in range(30):
        X_input = np.array(extended[-window_size:]).reshape(1, window_size, 1)
        pred = lstm_model.predict(X_input)[0, 0]
        forecast_date = last_date_forecast + pd.Timedelta(days=i+1)
        future_preds.append({"ds": forecast_date, "y_pred": pred})
        extended.append(pred)
    
    future_df = pd.DataFrame(future_preds)
    return df, future_df

# -----------------------------------------------------------------------------
# Step 4: Build the Streamlit Interface
# -----------------------------------------------------------------------------
st.title("Real-Time Soil Moisture Forecast for Latur Region, Maharashtra")
st.write("Fetching latest SMAP data and generating LSTM forecast (refreshes every day at 12 AM)...")

historical_df, forecast_df = get_forecast()

st.write("### Historical Soil Moisture Data")
st.line_chart(historical_df.set_index("ds")["y"])

st.write("### Forecasted Soil Moisture for Next 30 Days")
st.line_chart(forecast_df.set_index("ds")["y_pred"])

st.write("### Forecast Data Table")
st.dataframe(forecast_df)

if st.button("Refresh Forecast Now"):
    get_forecast.clear()
    historical_df, forecast_df = get_forecast()
    st.experimental_rerun()

# -----------------------------------------------------------------------------
# Step 5: Display Countdown Timer for Next Refresh with Transparent Background
# -----------------------------------------------------------------------------
countdown_html = """
<html>
  <body style="background-color: transparent; margin: 0; padding: 0;">
    <div id="countdown" style="font-size:20px; font-weight: bold; color: white; text-align: center;"></div>
    <script>
      var now = new Date();
      var tomorrow = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1);
      var distance = tomorrow - now;
      var x = setInterval(function() {
          var now = new Date().getTime();
          var distance = tomorrow - now;
          var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
          var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
          var seconds = Math.floor((distance % (1000 * 60)) / 1000);
          document.getElementById("countdown").innerHTML = "Next refresh at 12 AM in " + hours + "h " + minutes + "m " + seconds + "s";
          if (distance < 0) {
              clearInterval(x);
              document.getElementById("countdown").innerHTML = "Refreshing...";
          }
      }, 1000);
    </script>
  </body>
</html>
"""
components.html(countdown_html, height=100)
