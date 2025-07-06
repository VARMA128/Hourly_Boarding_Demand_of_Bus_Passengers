import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

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

# Define the generator model for the GAN
def build_generator(latent_dim, input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Dense(input_dim, activation='tanh'))
    return model

# Define the discriminator model for the GAN
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the GAN model combining the generator and discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

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

# Hyperparameters
latent_dim = 100
input_dim = X_train.shape[1]

# Build and compile the models
generator = build_generator(latent_dim, input_dim)
discriminator = build_discriminator(input_dim)
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Training the GAN
def train_gan(generator, discriminator, gan, X_train, epochs=10000, batch_size=64):
    batch_count = X_train.shape[0] // batch_size
    for epoch in range(epochs):
        for _ in range(batch_count):
            # Select a random batch of real data
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            
            # Generate fake data using the generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_data = generator.predict(noise)
            
            # Train the discriminator (real data is labeled 1, fake data is labeled 0)
            d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator (attempt to fool the discriminator)
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

# Train the GAN
train_gan(generator, discriminator, gan, X_train)

# Save the trained GAN models
generator.save("generator_model.h5")
discriminator.save("discriminator_model.h5")

print("GAN training complete and models saved.")
