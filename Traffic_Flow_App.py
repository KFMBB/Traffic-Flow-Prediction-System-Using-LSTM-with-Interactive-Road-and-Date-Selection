import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load models
@st.cache_resource
def load_models():
    models = {
        'King Fahad Road': tf.keras.models.load_model('Traffic_Models/King_Fahad_Road_Model.keras'),
        'King Khaled Road': tf.keras.models.load_model('Traffic_Models/King_Khaled_Road_Model.keras'),
        'King Salman Road': tf.keras.models.load_model('Traffic_Models/King_Salman_Road_Model.keras'),
        'Northern Ring Road': tf.keras.models.load_model('Traffic_Models/Northern_Ring_Road_Model.keras'),
        'King Abdullah Road': tf.keras.models.load_model('Traffic_Models/King_Abdullah_Road_Model.keras'),
    }
    return models

models = load_models()

# Prediction and classification
def predict_and_classify(road, time_series_data, scaler):
    model = models[road]
    prediction = model.predict(time_series_data)

    # Rescale the predictions if the data was scaled
    if scaler:
        prediction_rescaled = scaler.inverse_transform(prediction)
    else:
        prediction_rescaled = prediction

    return prediction_rescaled

# Function to load the data for the selected road
def load_data(road, window_hours=12):
    file_path = f'Roads_T5/{road.replace(" ", "_")}.csv'
    data = pd.read_csv(file_path, parse_dates=['timestamp'])

    # Define the current time as the latest available date in the data
    current_time = data['timestamp'].max()

    # Define the date range for filtering (window_hours before the current time)
    start_time = current_time - timedelta(hours=window_hours)
    
    # Filter data within the range of the prediction window
    filtered_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= current_time)]
    filtered_data = filtered_data.set_index('timestamp')

    return filtered_data, current_time

# Streamlit App Interface
st.title("Traffic Insights and Prediction Dashboard")

road = st.sidebar.selectbox(
    "Select a Road",
    list(models.keys())
)

data, current_time = load_data(road)

# Display the selected road and the current date range (last 12 hours)
st.write(f"### Road: {road}")
st.write(f"### Current Date: {datetime.now()}")

# Scale the data
scaler = MinMaxScaler()
time_series_data = scaler.fit_transform(data['hourly_traffic_count'].values.reshape(-1, 1))

if time_series_data.size < 12:  # Ensure there is enough data for prediction
    st.error("Error in prediction: Time series data is insufficient.")
else:
    prediction_rescaled = predict_and_classify(road, time_series_data, scaler)

    # Debugging information
    st.write("Prediction shape:", prediction_rescaled.shape)
    st.write("Prediction values:", prediction_rescaled.flatten())

    # Plot actual vs predicted traffic volume using Plotly
    st.write("#### Actual vs Predicted Traffic Volume")
    fig = go.Figure()

    prediction_time_index = data.index[-len(prediction_rescaled):]

    fig.add_trace(go.Scatter(x=data.index, y=data['hourly_traffic_count'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=prediction_time_index, y=prediction_rescaled.flatten(), mode='lines', name='Predicted'))

    fig.update_layout(
        title='Actual vs Predicted Traffic Volume',
        xaxis_title='Date',
        yaxis_title='Traffic Volume',
        xaxis=dict(tickformat='%Y-%m-%d %H:%M', tickangle=-45),
    )

    st.plotly_chart(fig)
