import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Loading the dataset
file_path = "Taxi Trip file.csv"  
data = pd.read_csv(file_path)

# Print the first few rows of the dataset
print("Initial Data Preview:")
print(data.head())

# Checking and handling missing values
print("\nMissing Values:")
print(data.isnull().sum())  

# Dropping rows with missing values
data = data.dropna()  

# Convert 'Time_of_Day' (categorical) to numeric values
time_mapping = {
    'Morning': 9,     # Represent Morning as 9 AM
    'Afternoon': 15,  # Represent Afternoon as 3 PM
    'Evening': 20,    # Represent Evening as 8 PM
    'Night': 23       # Represent Night as 11 PM
}
data['Time_of_Day'] = data['Time_of_Day'].map(time_mapping)  # Map the time of day to numeric

# Handled any invalid or missing values in 'Time_of_Day'
data = data.dropna(subset=['Time_of_Day'])  # Drop rows where 'Time_of_Day' is still missing or invalid

# Label encode the categorical columns
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns: 'Day_of_Week' and 'Traffic_Conditions'
data['Day_of_Week'] = label_encoder.fit_transform(data['Day_of_Week'])
data['Traffic_Conditions'] = label_encoder.fit_transform(data['Traffic_Conditions'])
data['Weather'] = label_encoder.fit_transform(data['Weather'])

# Print the first few rows after conversion
print("\nData after Label Encoding:")
print(data.head())

# Standardize numeric columns (distance and ride duration)
scaler = StandardScaler()
data[['Trip_Distance_km', 'Trip_Duration_Minutes']] = scaler.fit_transform(data[['Trip_Distance_km', 'Trip_Duration_Minutes']])

# Print the first few rows after scaling
print("\nData after Scaling:")
print(data[['Trip_Distance_km', 'Trip_Duration_Minutes']].head())

# Save the cleaned and transformed data to a new CSV file (optional)
data.to_csv("cleaned_taxi_trip_data.csv", index=False)

# --- Random Forest Regressor Model (Using All Features) ---
# Features (X) and target (y) for the new model (using all features)
X_new = data[['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count', 
              'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 
              'Per_Minute_Rate', 'Trip_Duration_Minutes']]  # All feature columns
y_new = data['Trip_Price']  # Target column (price)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

print(f"Training data size (New Model): {X_train.shape}")
print(f"Testing data size (New Model): {X_test.shape}")

# Initialize and train the Random Forest model
rf_model_new = RandomForestRegressor(random_state=42)
rf_model_new.fit(X_train, y_train)
rf_predictions_new = rf_model_new.predict(X_test)

# Evaluate the model
rf_r2_new = r2_score(y_test, rf_predictions_new)
rf_mae_new = mean_absolute_error(y_test, rf_predictions_new)






import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained Random Forest model
model = joblib.load('rf_model_new.pkl')

# Streamlit app layout
st.title("Taxi Fare Prediction App by Data Dynamos")



# Display an image for a more engaging experience
img_path = "WhatsApp Image 2025-01-24 at 14.17.42_98f67c99.jpg"  # Ensure this image path is correct
try:
    img = Image.open(img_path)
    st.image(img, caption="Welcome to the Taxi Fare Prediction App", use_column_width=True)
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
# For simplicity, we'll use some mock values or basic rules to simulate the automatic calculation of the background features.
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

# Make prediction if user clicks the "Predict Price" button
if st.button("Predict Price"):
    # Scale the input data to match the model's expected input
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform([features])  # scale the input features
    prediction = model.predict(scaled_features)

    # Display the predicted price
    st.write(f"The predicted taxi fare is: ${prediction[0]:.2f}")

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
