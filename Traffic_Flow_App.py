import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(filename='app.log', level=logging.INFO)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Traffic_Flow_LSTM_Model.h5')
    return model

model = load_model()

st.title("Traffic Flow Prediction System")

# Dropdown menu for selecting a road
road_list = [f"Road {i}" for i in range(1, 37)]  # Assuming roads are labeled as "Road 1" to "Road 36"
selected_road = st.selectbox("Select a Road", road_list)

# Input field for selecting a date
st.write("Please enter the date in the format YYYY-MM-DD.")
selected_date = st.text_input("Select a Date", placeholder="YYYY-MM-DD")

# Validate the date format (basic validation)
def validate_date(date_text):
    try:
        pd.to_datetime(date_text, format='%Y-%m-%d')
        return True
    except ValueError:
        return False

if selected_date and validate_date(selected_date):
    # Assuming you have a method to map the selected road and date to a CSV file
    def get_csv_for_road_and_date(road, date):
        # Example: Assuming the CSVs are named as 'Road_X_YYYY-MM-DD.csv'
        csv_filename = f"data/{road}_{date}.csv"
        if os.path.exists(csv_filename):
            return csv_filename
        else:
            st.error("Data not available for the selected road and date.")
            return None

    csv_file = get_csv_for_road_and_date(selected_road, selected_date)

    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("Preview of selected data:")
        st.dataframe(df.head())

        def preprocess_data(data):
            try:
                data = data.values
                data = data.reshape((1, data.shape[0], data.shape[1]))
                return data
            except Exception as e:
                logging.error(f"Error in preprocess_data: {e}")
                return None

        with st.spinner('Processing...'):
            processed_data = preprocess_data(df)
            if processed_data is not None:
                prediction = model.predict(processed_data)
                st.success(f"Predicted Traffic Flow: {prediction[0][0]:.2f}")
                logging.info(f"User selected {selected_road} on {selected_date}. Prediction: {prediction[0][0]:.2f}")
            else:
                st.error("Error in processing data.")
else:
    if selected_date:
        st.error("Please enter a valid date in the format YYYY-MM-DD.")

feedback = st.text_input("Provide feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
    logging.info(f"User feedback: {feedback}")
