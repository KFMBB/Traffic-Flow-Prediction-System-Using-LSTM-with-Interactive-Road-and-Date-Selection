# Traffic Flow Prediction Using LSTM

## Project Introduction

In this project, the goal is to develop a Long Short-Term Memory (LSTM) model to predict traffic flow across different roads. With increasing traffic congestion in urban areas, accurate traffic predictions can significantly enhance traffic management systems by optimizing routes and reducing congestion.

## Dataset Overview

The dataset for this project consists of historical traffic flow data collected from 36 different roads. Each road has its own set of CSV files, with traffic data recorded over time. This time-series data is crucial for understanding traffic patterns and making accurate predictions.

### Structure

- **Road Data**: The dataset includes traffic flow data indexed by road location and date.
- **Format**: Data is stored in CSV files, each corresponding to a specific road and date.

## Objective

The primary objective of this project is to build an LSTM model that can accurately predict traffic flow for a given road and date. This prediction can be used to anticipate traffic conditions, helping to manage traffic more effectively and reduce congestion.

## Methodology

### Data Preprocessing:

- **Time-series Analysis**: Organize and analyze the traffic flow data to identify patterns over time.
- **Normalization**: Normalize the data to ensure stable model training.
- **Reshape Data**: Format the data to fit the input requirements of the LSTM model.

### Model Architecture:

- **LSTM Layers**: Use LSTM layers to capture temporal dependencies in the traffic data.
- **Dense Layers**: Follow LSTM layers with dense layers to perform the final prediction.

### Training:

- **Data Splitting**: Split the dataset into training and validation sets.
- **Techniques**: Use techniques like early stopping and dropout to prevent overfitting and improve generalization.

### Evaluation:

- **Performance Metrics**: Evaluate the model’s performance using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Validation**: Use a validation set to monitor the model’s performance and adjust hyperparameters accordingly.

### Deployment:

- **Streamlit Application**: Deploy the trained model in a Streamlit app where users can select a road and input a date to get real-time traffic flow predictions.

## Data Source

The traffic flow data used in this project comes from a collection of CSV files, each representing traffic data for a specific road and date. These datasets are essential for training and evaluating the LSTM model.

## Streamlit App

Once the model is trained, it will be deployed using Streamlit, allowing users to interact with the model through a simple web interface. Users can select a road, input a date, and receive a traffic flow prediction.
Source: https://trafficflowforecast0bask.streamlit.app/
