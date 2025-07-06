from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash  # For password hashing
import sqlite3
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Initialize database
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            address TEXT,
            mobile TEXT,
            email TEXT UNIQUE NOT NULL,
            state TEXT,
            pincode TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load ML model and preprocessors
model = joblib.load("RandomForestClassifier_boarding_type_classification.joblib")
scaler = joblib.load("scaler.joblib")
encoders = joblib.load("encoders.joblib")
target_encoder = joblib.load("target_encoder.joblib")

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Register route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Get user input
        name = request.form["name"]
        username = request.form["username"]
        password = request.form["password"]
        address = request.form["address"]
        mobile = request.form["mobile"]
        email = request.form["email"]
        state = request.form["state"]
        pincode = request.form["pincode"]

        # Hash the password before storing
        hashed_password = generate_password_hash(password)

        # Check if the user already exists
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            flash("Username already exists. Please choose another one.", "danger")
            return render_template("register.html")

        # Add user to database
        cursor.execute("""
            INSERT OR IGNORE INTO users (name, username, password, address, mobile, email, state, pincode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, username, hashed_password, address, mobile, email, state, pincode))
        conn.commit()
        conn.close()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check if user exists in the database
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):  # User exists and password is correct
            session["username"] = username
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("predict"))  # Redirect to the predict page or home page after login
        else:
            flash("Invalid credentials. Please try again.", "danger")
            return render_template("login.html")

    return render_template("login.html")

# Helper function to preprocess input data
# Helper function to preprocess input data
def preprocess_input(data):
    encoded_data = {}
    for col, value in data.items():
        if col in encoders:
            if value in encoders[col].classes_:
                encoded_data[col] = encoders[col].transform([value])[0]
            else:
                fallback_value = encoders[col].classes_[0]
                encoded_data[col] = encoders[col].transform([fallback_value])[0]
        else:
            encoded_data[col] = value
    
    input_array = np.array([
        encoded_data.get("bus_route", 0),
        encoded_data.get("station_id", 0),
        encoded_data.get("weather_condition", 0),
        encoded_data.get("day_of_week", 0),
        encoded_data.get("card_type", 0),
        encoded_data.get("hourly_boarding_count", 0),
        encoded_data.get("holiday_flag", 0),
        data.get("hour", 0),
        data.get("day", 0),
        data.get("month", 0),
        data.get("weekday", 0),
    ]).reshape(1, -1)

    print("Preprocessed Input:", input_array)
    return scaler.transform(input_array)


    
  


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        flash("Please login to access this page.", "danger")
        return redirect(url_for("login"))

    if request.method == "POST":
        # Collect input data from the form
        bus_route = request.form["bus_route"]
        station_id = request.form["station_id"]
        weather_condition = request.form["weather_condition"]
        day_of_week = request.form["day_of_week"]
        card_type = request.form["card_type"]
        hourly_boarding_count = request.form["hourly_boarding_count"]
        holiday_flag = request.form["holiday_flag"]  # Get the holiday flag input

        # Extract the current timestamp (you can use current time or provide timestamp from the form)
        current_timestamp = pd.to_datetime("now")  # Adjust this if you have a timestamp input
        hour = current_timestamp.hour
        day = current_timestamp.day
        month = current_timestamp.month
        weekday = current_timestamp.weekday()

        # Prepare input data (including holiday_flag)
        input_data = {
            "bus_route": bus_route,
            "station_id": station_id,
            "weather_condition": weather_condition,
            "day_of_week": day_of_week,
            "card_type": card_type,
            "hourly_boarding_count": float(hourly_boarding_count),
            "holiday_flag": holiday_flag,  # Add holiday_flag to the input data
            "hour": hour,
            "day": day,
            "month": month,
            "weekday": weekday
        }

        # Preprocess the input data
        scaled_input = preprocess_input(input_data)

        # Get the model's prediction (you can select the model based on your need, e.g., RandomForestClassifier)
        #model = models["RandomForestClassifier"]  # Choose your model here
        prediction = model.predict(scaled_input)

        # Decode the predicted value
        predicted_boarding_type = target_encoder.inverse_transform(prediction)[0]

        # Return the result to the result page
        return render_template("result.html", result=predicted_boarding_type)

    return render_template("predict.html")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
