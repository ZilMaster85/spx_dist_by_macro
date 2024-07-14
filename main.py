import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from Excel file

data = pd.read_excel("df.xlsx")

# New vector
cpi_forecast = float(input("enter your cpi forecast: "))
gdp_forecast = float(input("enter your gdp forecast: "))
forecast_vector = np.array([cpi_forecast, gdp_forecast])  # Replace with your new 2D vector

# Parameter to adjust the number of closest points
num_closest_points = 50  # You can change this to 10, 100, etc.

# Step 1: Scale Column1 and Column2
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['cpi', 'gdp']])

# Scale the new vector using the same scaler
scaled_new_vector = scaler.transform(forecast_vector.reshape(1, -1))

# Step 2: Compute the Euclidean distance between the new vector and each point in the scaled dataset
data['Distance'] = np.sqrt((scaled_features[:, 0] - scaled_new_vector[0, 0])**2 + (scaled_features[:, 1] - scaled_new_vector[0, 1])**2)

# Step 3: Sort the data by distance
data_sorted_by_distance = data.sort_values(by='Distance').reset_index(drop=True)

# Step 4: Select the top N points with the smallest distances (N is defined by num_closest_points)
closest_points = data_sorted_by_distance.head(num_closest_points)

# Step 5: Get the values in the third column of all the closest points
closest_spx_values = closest_points['spx'].values

# Print the closest points and their corresponding Column3 values
print(f"Closest {num_closest_points} points:\n", closest_points)
print(f"Column3 values of the closest {num_closest_points} points:\n", closest_spx_values)

# Calculate median, mean, and standard deviation
median_value = np.median(closest_spx_values)
mean_value = np.mean(closest_spx_values)
std_deviation_value = np.std(closest_spx_values)

# Print the table of median, mean, and standard deviation
print(f"\nStatistics of the Column3 values of the closest {num_closest_points} points:")
print(f"Median: {median_value}")
print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_deviation_value}")

# Plot histogram
plt.hist(closest_spx_values, bins=10, edgecolor='black')
plt.title(f"Histogram of spx returns (Closest {num_closest_points} Points)")
plt.xlabel('Column3 Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
