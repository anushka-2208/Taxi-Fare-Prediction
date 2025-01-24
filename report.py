import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages

# Load the dataset
file_path = "Taxi Trip file.csv"
data = pd.read_csv(file_path)

# --- Data Preprocessing ---
# Handling Missing Values
data.dropna(inplace=True)

# Convert 'Time_of_Day' to numeric values
time_mapping = {'Morning': 9, 'Afternoon': 15, 'Evening': 20, 'Night': 23}
data['Time_of_Day'] = data['Time_of_Day'].map(time_mapping)

# Label Encoding
label_encoder = LabelEncoder()
data['Day_of_Week'] = label_encoder.fit_transform(data['Day_of_Week'])
data['Traffic_Conditions'] = label_encoder.fit_transform(data['Traffic_Conditions'])
data['Weather'] = label_encoder.fit_transform(data['Weather'])

# Standardize numeric features
scaler = StandardScaler()
data[['Trip_Distance_km', 'Trip_Duration_Minutes']] = scaler.fit_transform(data[['Trip_Distance_km', 'Trip_Duration_Minutes']])

# --- Exploratory Data Analysis (EDA) ---

# 1. Distribution of Trip Price
plt.figure(figsize=(8, 6))
sns.histplot(data['Trip_Price'], kde=True, color='green', bins=20)
plt.title('Distribution of Trip Prices')
plt.xlabel('Trip Price')
plt.ylabel('Frequency')
plt.savefig('trip_price_distribution.png')

# 2. Trip Price vs Distance (Scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Trip_Distance_km', y='Trip_Price', data=data, color='blue')
plt.title('Trip Price vs Distance')
plt.xlabel('Trip Distance (km)')
plt.ylabel('Trip Price')
plt.savefig('trip_price_vs_distance.png')

# --- Model Evaluation ---

# Features and target variable
X = data[['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count', 
          'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 
          'Per_Minute_Rate', 'Trip_Duration_Minutes']]
y = data['Trip_Price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

# Save evaluation metrics to the report
metrics = f"Random Forest Model Evaluation:\nR²: {rf_r2}\nMAE: {rf_mae}\n\n"

# --- Generating the Report PDF ---

with PdfPages('model_report.pdf') as pdf:
    # Title Page
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, 'Taxi Trip Price Prediction Report', fontsize=20, ha='center', va='center')
    plt.axis('off')
    pdf.savefig()
    plt.close()

    # Data Preprocessing Section
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, 'Data Preprocessing: Missing Values Handling, Encoding, and Scaling', fontsize=14, ha='center', va='center')
    plt.axis('off')
    pdf.savefig()
    plt.close()

    # Add the feature importance (or any other result you want to include)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=rf_model.feature_importances_, y=X.columns, palette="Blues_d")
    plt.title('Feature Importance for Trip Price Prediction')
    plt.savefig('feature_importance.png')
    pdf.savefig()
    plt.close()

    # EDA Plots
    plt.figure(figsize=(8, 6))
    plt.imshow(plt.imread('trip_price_distribution.png'))
    plt.axis('off')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(plt.imread('trip_price_vs_distance.png'))
    plt.axis('off')
    pdf.savefig()
    plt.close()

    # Model Evaluation
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, metrics, fontsize=12, ha='center', va='center')
    plt.axis('off')
    pdf.savefig()
    plt.close()
