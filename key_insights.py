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



import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Update the 'Time_of_Day' back to categorical labels
time_mapping = {9: 'Morning', 15: 'Afternoon', 20: 'Evening', 23: 'Night'}
data['Time_of_Day'] = data['Time_of_Day'].map(time_mapping)

# 1. Trip Price vs. Trip Distance (Scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Trip_Distance_km', y='Trip_Price', data=data, color='blue', alpha=0.6)
plt.title('Trip Price vs Trip Distance (km)', fontsize=14)
plt.xlabel('Trip Distance (km)')
plt.ylabel('Trip Price')
plt.show()

# 2. Trip Price by Time of Day (Count plot)
plt.figure(figsize=(8, 6))
sns.countplot(x='Time_of_Day', data=data, palette="Set2")
plt.title('Count of Trips by Time of Day', fontsize=14)
plt.xlabel('Time of Day')
plt.ylabel('Count of Trips')
plt.show()

# 3. Trip Price by Day of the Week (Bar plot)
plt.figure(figsize=(8, 6))
sns.barplot(x='Day_of_Week', y='Trip_Price', data=data, palette="muted")
plt.title('Average Trip Price by Day of the Week', fontsize=14)
plt.xlabel('Day of the Week')
plt.ylabel('Average Trip Price')
plt.show()



# 6. Trip Price by Weather (Pie chart)
weather_counts = data['Weather'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Pastel1"))
plt.title('Trip Price Distribution by Weather', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# Line chart for Average Trip Price vs Passenger Count
plt.figure(figsize=(8, 6))
sns.lineplot(x='Passenger_Count', y='Trip_Price', data=data, marker='o')
plt.title('Average Trip Price by Passenger Count (Line Chart)', fontsize=14)
plt.xlabel('Passenger Count')
plt.ylabel('Average Trip Price')
plt.show()




