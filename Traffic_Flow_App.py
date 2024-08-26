import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

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

def calculate_traffic_status(predictions):
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(predictions.reshape(-1, 1))
    thresholds = sorted(kmeans.cluster_centers_.flatten())
    return thresholds

def predict_and_classify(road, x):
    model = models[road]
    y_pred = model.predict(x)
    
    thresholds = calculate_traffic_status(y_pred)

    if y_pred[-1] <= thresholds[0]:
        traffic_status = "Free Flow"
    elif y_pred[-1] <= thresholds[1]:
        traffic_status = "Light Traffic"
    elif y_pred[-1] <= thresholds[2]:
        traffic_status = "Moderate Traffic"
    elif y_pred[-1] <= thresholds[3]:
        traffic_status = "Heavy Traffic"
    else:
        traffic_status = "Severe Congestion"

    return y_pred, traffic_status

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

st.title("Traffic Insights and Prediction Dashboard")

road = st.sidebar.selectbox(
    "Select a Road",
    list(models.keys())
)

data, current_time = load_data(road)

# Display the selected road and the current date range (last 12 hours)
st.write(f"### Road: {road}")
st.write(f"### Current Date: {datetime.now()}")

# Predict traffic volume for the selected road
x = data['hourly_traffic_count'].values.reshape(-1, 1)  # x is the input features
if x.size < 12:  # Ensure there is enough data for prediction
    st.error("Error in prediction: Time series data is insufficient.")
else:
    y_pred, traffic_status = predict_and_classify(road, x)  # y_pred is the predicted output

    # Display prediction and traffic status
    st.write(f"#### Predicted Traffic Volume: {y_pred[-1][0]:.2f}")
    st.write(f"#### Traffic Status: {traffic_status}")

    # Plot actual vs predicted traffic volume using Plotly
    st.write("#### Actual vs Predicted Traffic Volume")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['hourly_traffic_count'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data.index, y=np.concatenate(y_pred), mode='lines', name='Predicted'))

    fig.update_layout(
        title='Actual vs Predicted Traffic Volume',
        xaxis_title='Date',
        yaxis_title='Traffic Volume',
        xaxis=dict(tickformat='%Y-%m-%d %H:%M', tickangle=-45),
    )

    st.plotly_chart(fig)

    # Display metrics
    st.write("#### Traffic Insights and Metrics")
    avg_actual = data['hourly_traffic_count'].mean()
    avg_predicted = np.concatenate(y_pred).mean()
    peak_actual = data['hourly_traffic_count'].max()
    peak_predicted = np.concatenate(y_pred).max()

    st.metric("Average hourly_traffic_count", f"{avg_actual:.2f}")
    st.metric("Average Predicted Traffic Volume", f"{avg_predicted:.2f}")
    st.metric("Peak hourly_traffic_count", f"{peak_actual}")
    st.metric("Peak Predicted Traffic Volume", f"{peak_predicted}")

    # Display prediction error analysis using Plotly
    st.write("#### Prediction Error Analysis")
    data['Predicted Traffic Volume'] = np.concatenate(y_pred)
    data['Error'] = data['hourly_traffic_count'] - data['Predicted Traffic Volume']
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=data['Error'], nbinsx=50, histfunc='count', name='Error Distribution'))

    fig.update_layout(
        title='Prediction Error Distribution',
        xaxis_title='Error (Actual - Predicted)',
        yaxis_title='Frequency'
    )

    st.plotly_chart(fig)

    # Model performance summary
    st.write("#### Model Performance Summary")
    mae = np.mean(np.abs(data['Error']))
    mse = np.mean(data['Error'] ** 2)
    rmse = np.sqrt(mse)

    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
