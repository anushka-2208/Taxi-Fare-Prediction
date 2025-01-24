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

# Print performance metrics
print(f"Random Forest R²: {rf_r2_new}")
print(f"Random Forest MAE: {rf_mae_new}")

# Visualization of Actual vs Predicted Prices
import matplotlib.pyplot as plt

plt.scatter(y_test, rf_predictions_new, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.show()

# Save the trained model (optional)
import joblib
joblib.dump(rf_model_new, 'rf_model_new.pkl')
