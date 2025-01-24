import pandas as pd
from sklearn.preprocessing import StandardScaler

#Loading the dataset
file_path = "Taxi Trip file.csv"  
data = pd.read_csv(file_path)

# Print the first few rows of the dataset
print("Initial Data Preview:")
print(data.head())

#Checking and handling missing values
print("\nMissing Values:")
print(data.isnull().sum())  

#Dropping rows with missing values
data = data.dropna()  

#Convert 'Time_of_Day' (categorical) to numeric values
time_mapping = {
    'Morning': 9,     # Represent Morning as 9 AM
    'Afternoon': 15,  # Represent Afternoon as 3 PM
    'Evening': 20,    # Represent Evening as 8 PM
    'Night': 23       # Represent Night as 11 PM
}
data['Time_of_Day'] = data['Time_of_Day'].map(time_mapping)  # Map the time of day to numeric

# Handled any invalid or missing values in 'Time_of_Day'
data = data.dropna(subset=['Time_of_Day'])  # Drop rows where 'Time_of_Day' is still missing or invalid

# Print the first few rows after conversion
print("\nData after Time_of_Day Conversion:")
print(data.head())

#Standardize numeric columns (distance and ride duration)

scaler = StandardScaler()
data[['Trip_Distance_km', 'Trip_Duration_Minutes']] = scaler.fit_transform(data[['Trip_Distance_km', 'Trip_Duration_Minutes']])

# Print the first few rows after scaling
print("\nData after Scaling:")
print(data[['Trip_Distance_km', 'Trip_Duration_Minutes']].head())

# Save the cleaned and transformed data to a new CSV file (optional)
data.to_csv("cleaned_taxi_trip_data.csv", index=False)


from sklearn.model_selection import train_test_split

# Features (X) and target (y)
X = data[['Trip_Distance_km', 'Time_of_Day', 'Trip_Duration_Minutes']]  # Feature columns
y = data['Trip_Price']  # Target column (price)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")


from sklearn.linear_model import LinearRegression

# Initialize and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
lr_predictions = lr_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import r2_score, mean_absolute_error

lr_r2 = r2_score(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)

print(f"Linear Regression R²: {lr_r2}")
print(f"Linear Regression MAE: {lr_mae}")


from sklearn.ensemble import RandomForestRegressor

# Initialize and train the random forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

print(f"Random Forest R²: {rf_r2}")
print(f"Random Forest MAE: {rf_mae}")


import matplotlib.pyplot as plt

# Plot actual vs predicted prices for the best model (Random Forest or Linear Regression)
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.show()


import seaborn as sns

# Price vs Distance
sns.scatterplot(x='Trip_Distance_km', y='Trip_Price', data=data)
plt.title('Price vs Distance')
plt.show()


# Average Price by Time of Day
sns.barplot(x='Time_of_Day', y='Trip_Price', data=data)
plt.title('Average Price by Time of Day')
plt.show()


import streamlit as st
import pandas as pd
import numpy as np

# Load the trained model (Random Forest or Linear Regression)
model = rf_model  # Assuming Random Forest model was chosen

# Web app title
st.title("Taxi Price Prediction")

# Input fields for user to enter ride details
distance = st.number_input("Enter Distance (in km):", min_value=0.0, step=0.1)
time_of_day = st.selectbox("Select Time of Day:", ['Morning', 'Afternoon', 'Evening', 'Night'])
duration = st.number_input("Enter Ride Duration (in minutes):", min_value=0.0, step=0.1)

# Convert the time_of_day to numeric using the same mapping
time_mapping = {'Morning': 9, 'Afternoon': 15, 'Evening': 20, 'Night': 23}
time_of_day_numeric = time_mapping[time_of_day]

# Predict price
if st.button("Predict Price"):
    prediction = model.predict([[distance, time_of_day_numeric, duration]])
    st.write(f"Predicted Price: ${prediction[0]:.2f}")
