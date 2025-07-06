import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Preprocessing function with feature engineering
def preprocess_data(data, target_column='boarding_type'):
    # Extract hour, day, and month from timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['weekday'] = data['timestamp'].dt.weekday

    # Drop irrelevant columns
    data = data.drop(['record_id', 'timestamp'], axis=1)

    # Encode categorical features (excluding target column)
    categorical_columns = ['bus_route', 'station_id', 'weather_condition', 'day_of_week', 'card_type']
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    # Encode target column
    target_encoder = LabelEncoder()
    data[target_column] = target_encoder.fit_transform(data[target_column])

    # Select features and target
    features = data.drop([target_column], axis=1)
    target = data[target_column]

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, scaler, encoders, target_encoder

# Load dataset
file_path = "expanded_bus_boarding_demand_dataset.csv"
data = pd.read_csv(file_path)

# Preprocess the dataset
X, y, scaler, encoders, target_encoder = preprocess_data(data, target_column='boarding_type')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification models
models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "XGBClassifier": XGBClassifier(random_state=42)
}

# Train and save models
model_paths = {}
for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = f"{model_name}_boarding_type_classification.joblib"
    joblib.dump(model, model_path)
    model_paths[model_name] = model_path

# Save preprocessors
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoders, "encoders.joblib")
joblib.dump(target_encoder, "target_encoder.joblib")

# Print model paths
print("Saved model paths:", model_paths)
print("Saved preprocessing objects: scaler.joblib, encoders.joblib, target_encoder.joblib")
