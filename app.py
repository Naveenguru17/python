from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Define the available drug types
drugs = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

# Define the pattern for different frequency model files
model_file_patterns = {

    'daily':'auto_arima_model_daily_{drug}.pkl',
    'weekly': 'auto_arima_model_week_{drug}.pkl',
    'monthly': 'auto_arima_model_{drug}.pkl'

}

# Function to load the appropriate model based on the frequency and drug type
def load_model(freq, drug):
    model_file = model_file_patterns[freq].format(drug=drug)
    with open(model_file, 'rb') as file:
        return pickle.load(file)

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    date = data['date']
    prediction_type = data['type']

    # Define the appropriate date range and frequency based on prediction type
    freq_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'yearly': 'Y'
    }

    # Convert the date string to a pandas Timestamp
    target_date = pd.to_datetime(date)

    # Prepare the date range for prediction
    start_date = '2019-11-30'  # Start date for all models
    date_range = pd.date_range(start=start_date, end=target_date, freq=freq_map[prediction_type])

    predictions = {}
    for drug in drugs:
        # Load the appropriate model
        model = load_model(prediction_type, drug)
        
        # Make predictions up to the target date
        predicted_values = model.predict(n_periods=len(date_range))
        
        # Extract the prediction for the specific date (end of the range)
        predictions[drug] = predicted_values[-1]

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(port=5000)