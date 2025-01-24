import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# --- Existing Model (Limited Features) ---
# Features (X) and target (y) for the existing model (using limited features)
X_existing = data[['Trip_Distance_km', 'Time_of_Day', 'Trip_Duration_Minutes']]  # Limited feature columns
y_existing = data['Trip_Price']  # Target column (price)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_existing, y_existing, test_size=0.2, random_state=42)

print(f"Training data size (Existing Model): {X_train.shape}")
print(f"Testing data size (Existing Model): {X_test.shape}")

# --- Linear Regression Model (Existing Features) ---
lr_model_existing = LinearRegression()
lr_model_existing.fit(X_train, y_train)
lr_predictions_existing = lr_model_existing.predict(X_test)

# Evaluate the model
lr_r2_existing = r2_score(y_test, lr_predictions_existing)
lr_mae_existing = mean_absolute_error(y_test, lr_predictions_existing)

# --- Random Forest Regressor Model (Existing Features) ---
rf_model_existing = RandomForestRegressor(random_state=42)
rf_model_existing.fit(X_train, y_train)
rf_predictions_existing = rf_model_existing.predict(X_test)

# Evaluate the model
rf_r2_existing = r2_score(y_test, rf_predictions_existing)
rf_mae_existing = mean_absolute_error(y_test, rf_predictions_existing)

# --- New Model (With All Features) ---
# Features (X) and target (y) for the new model (using all features)
X_new = data[['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count', 
              'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 
              'Per_Minute_Rate', 'Trip_Duration_Minutes']]  # All feature columns
y_new = data['Trip_Price']  # Target column (price)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

print(f"Training data size (New Model): {X_train.shape}")
print(f"Testing data size (New Model): {X_test.shape}")

# --- Linear Regression Model (New Features) ---
lr_model_new = LinearRegression()
lr_model_new.fit(X_train, y_train)
lr_predictions_new = lr_model_new.predict(X_test)

# Evaluate the model
lr_r2_new = r2_score(y_test, lr_predictions_new)
lr_mae_new = mean_absolute_error(y_test, lr_predictions_new)

# --- Random Forest Regressor Model (New Features) ---
rf_model_new = RandomForestRegressor(random_state=42)
rf_model_new.fit(X_train, y_train)
rf_predictions_new = rf_model_new.predict(X_test)

# Evaluate the model
rf_r2_new = r2_score(y_test, rf_predictions_new)
rf_mae_new = mean_absolute_error(y_test, rf_predictions_new)

# --- Comparison of Models (Existing vs New) ---

# Linear Regression Comparison
print("\nComparison of Linear Regression Models:")
print(f"Existing Model Linear Regression R²: {lr_r2_existing}")
print(f"Existing Model Linear Regression MAE: {lr_mae_existing}")
print(f"New Model Linear Regression R²: {lr_r2_new}")
print(f"New Model Linear Regression MAE: {lr_mae_new}")
print(f"Linear Regression R² improvement: {lr_r2_new - lr_r2_existing}")
print(f"Linear Regression MAE improvement: {lr_mae_existing - lr_mae_new}")

# Random Forest Comparison
print("\nComparison of Random Forest Models:")
print(f"Existing Model Random Forest R²: {rf_r2_existing}")
print(f"Existing Model Random Forest MAE: {rf_mae_existing}")
print(f"New Model Random Forest R²: {rf_r2_new}")
print(f"New Model Random Forest MAE: {rf_mae_new}")
print(f"Random Forest R² improvement: {rf_r2_new - rf_r2_existing}")
print(f"Random Forest MAE improvement: {rf_mae_existing - rf_mae_new}")



import matplotlib.pyplot as plt
import seaborn as sns

model_names = ['Linear Regressor (Selective Features)', 'Random Forest Regressor (Selective Features)', 
               'Linear Regressor (All Features)', 'Random Forest Regressor (All Features)']
r2_values = [lr_r2_existing, rf_r2_existing, lr_r2_new, rf_r2_new]
mae_values = [lr_mae_existing, rf_mae_existing, lr_mae_new, rf_mae_new]

# Create the figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for R² values
ax1.bar(model_names, r2_values, color='royalblue', alpha=0.6, label="R²")

# Create another y-axis for MAE values
ax2 = ax1.twinx()
ax2.bar(model_names, mae_values, color='darkorange', alpha=0.6, label="MAE")

# Labels and title
ax1.set_ylabel("R² (Model Fit)", color='royalblue')
ax2.set_ylabel("MAE (Error)", color='darkorange')
ax1.set_title('Comparison of Model Performance (R² & MAE)', fontsize=14)

# Show the plot
fig.tight_layout()
plt.show()