import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load the trained Random Forest model
model = joblib.load('rf_model_new.pkl')

# Streamlit app layout
st.title("Taxi Fare Prediction App by Data Dynamos")

# Display an image for a more engaging experience
img_path = "WhatsApp Image 2025-01-24 at 14.17.42_98f67c99.jpg"  # Ensure this image path is correct
try:
    img = Image.open(img_path)
    st.image(img, caption="Welcome to the Taxi Fare Prediction App", use_container_width=True)
except FileNotFoundError:
    st.write("Image not found. Please ensure the 'taxi_image.jpg' file is in the same directory.")

# Create the input fields for user input
st.header("Enter Trip Details")

# Input fields for user to enter data
trip_distance = st.number_input("Enter Trip Distance (in km):", min_value=0.1, step=0.1)
time_of_day = st.selectbox("Select Time of Day:", ['Morning', 'Afternoon', 'Evening', 'Night'])
passenger_count = st.number_input("Enter Number of Passengers:", min_value=1, step=1)
trip_duration = st.number_input("Enter Trip Duration (in minutes):", min_value=1, step=1)

# Mapping the time of day to numeric value for the model
time_mapping = {'Morning': 9, 'Afternoon': 15, 'Evening': 20, 'Night': 23}
time_of_day_numeric = time_mapping[time_of_day]

# Creating background feature calculation
def get_background_features():
    # You can replace this with any logic or feature calculation you want.
    day_of_week = np.random.choice([0, 1, 2, 3, 4, 5, 6])  # Random day of the week (0=Monday, 6=Sunday)
    traffic_condition = np.random.choice([0, 1, 2])  # Random traffic condition (0=Low, 1=Medium, 2=High)
    weather_condition = np.random.choice([0, 1, 2])  # Random weather condition (0=Clear, 1=Rainy, 2=Foggy)
    base_fare = 3.00  # Just a sample base fare
    per_km_rate = 1.5  # Sample per km rate
    per_min_rate = 0.2  # Sample per minute rate

    return day_of_week, traffic_condition, weather_condition, base_fare, per_km_rate, per_min_rate

# Get the background features
day_of_week, traffic_condition, weather_condition, base_fare, per_km_rate, per_min_rate = get_background_features()

# Display these features for the user (if needed)
st.write(f"Automatically Calculated Background Features:")
st.write(f"Day of Week: {day_of_week}, Traffic Condition: {traffic_condition}, Weather: {weather_condition}")
st.write(f"Base Fare: {base_fare}, Per Km Rate: {per_km_rate}, Per Minute Rate: {per_min_rate}")

# Prepare the data for prediction (background features + user inputs)
features = [
    trip_distance,
    time_of_day_numeric,
    day_of_week,
    passenger_count,
    traffic_condition,
    weather_condition,
    base_fare,
    per_km_rate,
    per_min_rate,
    trip_duration
]

# Conversion rate from USD to INR (example rate, adjust accordingly)
USD_to_INR = 82  # Example conversion rate, can be updated with real-time exchange rate

# Make prediction if user clicks the "Predict Price" button
if st.button("Predict Price"):
    # Scale the input data to match the model's expected input
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform([features])  # scale the input features
    prediction_usd = model.predict(scaled_features)

    # Convert the predicted fare from USD to INR
    prediction_inr = prediction_usd[0] * USD_to_INR

    # Display the predicted price in INR
    st.write(f"The predicted taxi fare is: ₹{prediction_inr:.2f}")

# Show a simple plot for user interaction (optional)
st.header("Trip Price vs Distance Plot")
# Plotting the relationship between trip distance and price (for example)
distance_range = np.linspace(1, 50, 50)  # Generate a range of trip distances from 1 km to 50 km
# Ensure scaler is defined before applying it
scaler = StandardScaler()
price_predictions = model.predict(scaler.fit_transform(np.column_stack([distance_range, np.zeros(50), np.zeros(50), np.ones(50), np.zeros(50), np.zeros(50), np.ones(50), np.ones(50), np.ones(50), np.ones(50)])))

plt.figure(figsize=(8, 6))
plt.plot(distance_range, price_predictions, label="Predicted Price", color="blue")
plt.xlabel("Trip Distance (km)")
plt.ylabel("Predicted Price ($)")
plt.title("Taxi Fare Prediction: Price vs Distance")
plt.legend()
st.pyplot(plt)
