import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime, timedelta


# Load Models for Each Road
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


# Set thresholds for classification using KMeans
def calculate_traffic_status(predictions):
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(predictions.reshape(-1, 1))
    thresholds = sorted(kmeans.cluster_centers_.flatten())
    return thresholds


# Predict Traffic Volume and Classify Status
def predict_and_classify(road, time_series_data):
    model = models[road]
    prediction = model.predict(time_series_data)

    thresholds = calculate_traffic_status(prediction)

    if prediction <= thresholds[0]:
        traffic_status = "Very Low Traffic"
    elif prediction <= thresholds[1]:
        traffic_status = "Low Traffic"
    elif prediction <= thresholds[2]:
        traffic_status = "Moderate Traffic"
    elif prediction <= thresholds[3]:
        traffic_status = "High Traffic"
    else:
        traffic_status = "Very High Traffic"

    return prediction, traffic_status


# Function to load the data for the selected road
def load_data(road, current_time):
    file_path = f'Roads_T5/{road.replace(" ", "_")}.csv'
    data = pd.read_csv(file_path, parse_dates=['Date'])

    # Filter data within the range of 2 hours before the current time
    start_time = current_time - timedelta(hours=2)
    data = data[(data['Date'] >= start_time) & (data['Date'] <= current_time)]
    data = data.set_index('Date')

    return data


# Streamlit App
st.title("Traffic Insights and Prediction Dashboard")

# Sidebar for road selection
road = st.sidebar.selectbox(
    "Select a Road",
    list(models.keys())
)

# Get the user's current time
current_time = datetime.now()

# Load the data based on current time
data = load_data(road, current_time)

# Display the selected road and the current date range (last 2 hours)
st.write(f"### Road: {road}")
st.write(f"### Date Range: {current_time - timedelta(hours=2)} to {current_time}")

# Predict traffic volume for the selected road
time_series_data = data['Actual Traffic Volume'].values.reshape(-1, 1)
prediction, traffic_status = predict_and_classify(road, time_series_data)

# Display prediction and traffic status
st.write(f"#### Predicted Traffic Volume: {prediction[-1][0]:.2f}")
st.write(f"#### Traffic Status: {traffic_status}")

# Plot actual vs predicted traffic volume
st.write("#### Actual vs Predicted Traffic Volume")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=data, x=data.index, y='Actual Traffic Volume', label='Actual', ax=ax)
sns.lineplot(data=data, x=data.index, y='Predicted Traffic Volume', label='Predicted', ax=ax)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Display metrics
st.write("#### Traffic Insights and Metrics")
avg_actual = data['Actual Traffic Volume'].mean()
avg_predicted = data['Predicted Traffic Volume'].mean()
peak_actual = data['Actual Traffic Volume'].max()
peak_predicted = data['Predicted Traffic Volume'].max()

st.metric("Average Actual Traffic Volume", f"{avg_actual:.2f}")
st.metric("Average Predicted Traffic Volume", f"{avg_predicted:.2f}")
st.metric("Peak Actual Traffic Volume", f"{peak_actual}")
st.metric("Peak Predicted Traffic Volume", f"{peak_predicted}")

# Display prediction error analysis
st.write("#### Prediction Error Analysis")
data['Error'] = data['Actual Traffic Volume'] - data['Predicted Traffic Volume']
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['Error'], kde=True, ax=ax)
plt.title("Prediction Error Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
st.pyplot(fig)

# Model performance summary
st.write("#### Model Performance Summary")
mae = np.mean(np.abs(data['Error']))
mse = np.mean(data['Error'] ** 2)
rmse = np.sqrt(mse)

st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
