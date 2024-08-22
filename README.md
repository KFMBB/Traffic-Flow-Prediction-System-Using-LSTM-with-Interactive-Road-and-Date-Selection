# Traffic Flow Prediction System Using LSTM with Interactive Road and Date Selection

## Overview

This project aims to predict traffic flow on different roads using a Long Short-Term Memory (LSTM) model. The system allows users to interactively select a specific road and date, then provides a traffic flow prediction based on historical data. The model is deployed as a web application using Streamlit.

## Features

- **Road Selection**: Users can choose from 36 different roads for traffic flow prediction.
- **Date Input**: Users can specify a date for prediction in the format `YYYY-MM-DD`.
- **Real-time Prediction**: The application processes the selected data and outputs the predicted traffic flow.
- **Interactive Interface**: The web application is built using Streamlit, offering a simple and user-friendly interface.
- **Logging**: User interactions and system events are logged for monitoring and debugging purposes.

## Installation

### Prerequisites

- Python 3.7+
- Pip (Python package installer)

### Step-by-Step Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/traffic-flow-prediction.git
    cd traffic-flow-prediction
    ```

2. **Create a Virtual Environment (optional but recommended)**:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Model and Data**:
    - Place the pre-trained LSTM model `Traffic_Flow_LSTM_Model.h5` in the project root directory.
    - Organize your CSV files in a directory named `data/`, following the naming convention `Road_X_YYYY-MM-DD.csv`.

5. **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the Streamlit application in your web browser.
2. Select a road from the dropdown menu.
3. Enter a date in the format `YYYY-MM-DD`.
4. Click the button to receive the predicted traffic flow for the selected road and date.
5. View the prediction result, including the predicted traffic flow value.

## Data

The application uses historical traffic flow data stored in CSV files. Each file corresponds to a specific road and date. Ensure that the CSV files are named according to the format `Road_X_YYYY-MM-DD.csv` and placed in the `data/` directory.

## Model

The LSTM model is trained on time-series traffic data, designed to capture patterns and trends over time. The model is loaded from the `Traffic_Flow_LSTM_Model.h5` file and used to make predictions based on the selected road and date.

## Logging

The application logs user interactions and system events to a file named `app.log`. This is useful for monitoring application usage and diagnosing issues.

## Feedback

Users can provide feedback directly through the application interface. This feedback is logged for further analysis.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to follow the project's coding guidelines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or suggestions, please contact [your email address].
