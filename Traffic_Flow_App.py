import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load Models for Each Road
@st.cache_resource
def load_models():
    models = {
        'King Fahad Road': tf.keras.models.load_model('King_Fahad_Road_Model.keras'),
        'King Khaled Road': tf.keras.models.load_model('King_Khaled_Road_Model.keras'),
        'King Salman Road': tf.keras.models.load_model('King_Salman_Road_Model.keras'),
        'Northern Ring Road': tf.keras.models.load_model('Northern_Ring_Road_Model.keras'),
        'King Abdullah Road': tf.keras.models.load_model('King_Abdullah_Road_Model.keras'),
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
def load_data(road, start_date, end_date):
    # Placeholder function to simulate data loading
    # Replace with actual data loading code
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    data = pd.DataFrame({
        'Date': dates,
        'Actual Traffic Volume': np.random.randint(100, 500, len(dates)),
        'Predicted Traffic Volume': np.random.randint(100, 500, len(dates))
    })
    return data

# Streamlit App
st.title("Traffic Insights and Prediction Dashboard")

# Sidebar for road selection
road = st.sidebar.selectbox(
    "Select a Road",
    list(models.keys())
)

# Date range selection
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Load the data
data = load_data(road, start_date, end_date)

# Display the selected road and date range
st.write(f"### Road: {road}")
st.write(f"### Date Range: {start_date} to {end_date}")

# Predict traffic volume for the selected road and date range
time_series_data = data['Actual Traffic Volume'].values.reshape(-1, 1)
prediction, traffic_status = predict_and_classify(road, time_series_data)

# Display prediction and traffic status
st.write(f"#### Predicted Traffic Volume: {prediction[0][0]:.2f}")
st.write(f"#### Traffic Status: {traffic_status}")

# Plot actual vs predicted traffic volume
st.write("#### Actual vs Predicted Traffic Volume")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=data, x='Date', y='Actual Traffic Volume', label='Actual', ax=ax)
sns.lineplot(data=data, x='Date', y='Predicted Traffic Volume', label='Predicted', ax=ax)
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
mse = np.mean(data['Error']**2)
rmse = np.sqrt(mse)

st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
