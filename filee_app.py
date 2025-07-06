import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained models and preprocessors
scaler = joblib.load("scaler.joblib")
encoders = joblib.load("encoders.joblib")
target_encoder = joblib.load("target_encoder.joblib")

logistic_model = joblib.load("LogisticRegression_boarding_type_classification.joblib")
rf_model = joblib.load("RandomForestClassifier_boarding_type_classification.joblib")
xgb_model = joblib.load("XGBClassifier_boarding_type_classification.joblib")

# Load the GAN models (generator and discriminator)
generator = load_model("generator_model.h5")
discriminator = load_model("discriminator_model.h5")

# Preprocessing function for input data
def preprocess_input(input_data):
    input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
    input_data['hour'] = input_data['timestamp'].dt.hour
    input_data['day'] = input_data['timestamp'].dt.day
    input_data['month'] = input_data['timestamp'].dt.month
    input_data['weekday'] = input_data['timestamp'].dt.weekday
    
    # Drop irrelevant columns
    input_data = input_data.drop(['record_id', 'timestamp'], axis=1)
    
    # Encode categorical features
    for col in ['bus_route', 'station_id', 'weather_condition', 'day_of_week', 'card_type']:
        encoder = encoders[col]
        input_data[col] = encoder.transform(input_data[col])
    
    # Scale the features
    features = input_data.drop(['boarding_type'], axis=1)
    scaled_features = scaler.transform(features)
    
    return scaled_features

# Function to generate synthetic data using the GAN generator
def generate_synthetic_data(latent_dim=100, num_samples=100):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_data = generator.predict(noise)
    return synthetic_data

# Define the route to classify boarding type using the classification models
@app.route('/classify_boarding', methods=['POST'])
def classify_boarding():
    data = request.get_json()  # Assuming input is JSON
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([data])
    
    # Preprocess input
    preprocessed_input = preprocess_input(input_df)
    
    # Get predictions from all classification models
    logistic_pred = logistic_model.predict(preprocessed_input)
    rf_pred = rf_model.predict(preprocessed_input)
    xgb_pred = xgb_model.predict(preprocessed_input)
    
    # Decode the predictions
    decoded_logistic_pred = target_encoder.inverse_transform(logistic_pred)
    decoded_rf_pred = target_encoder.inverse_transform(rf_pred)
    decoded_xgb_pred = target_encoder.inverse_transform(xgb_pred)
    
    response = {
        "LogisticRegression_Prediction": decoded_logistic_pred[0],
        "RandomForest_Prediction": decoded_rf_pred[0],
        "XGBoost_Prediction": decoded_xgb_pred[0]
    }
    
    return jsonify(response)

# Define the route to generate synthetic data using GAN
@app.route('/generate_synthetic_data', methods=['POST'])
def generate_data():
    data = request.get_json()  # Assuming input is JSON
    
    # Number of synthetic samples to generate (optional input)
    num_samples = data.get("num_samples", 100)
    
    # Generate synthetic data using the GAN
    synthetic_data = generate_synthetic_data(num_samples=num_samples)
    
    # Convert synthetic data to a DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=['bus_route', 'station_id', 'weather_condition', 'hour', 'day', 'month', 'weekday', 'card_type'])
    
    # Encode categorical columns
    for col in ['bus_route', 'station_id', 'weather_condition', 'card_type']:
        encoder = encoders[col]
        synthetic_df[col] = encoder.inverse_transform(synthetic_df[col].astype(int))
    
    # Return synthetic data as a JSON response
    synthetic_data_json = synthetic_df.to_dict(orient='records')
    return jsonify(synthetic_data_json)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
