import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from Excel file
data = pd.read_excel("df.xlsx")

# Input forecast vector
cpi_forecast = float(input("Enter your CPI forecast: "))
gdp_forecast = float(input("Enter your GDP forecast: "))
forecast_vector = np.array([cpi_forecast, gdp_forecast])

# Parameter to adjust the number of closest points
num_closest_points = 30 # cross validation reduces the error a lot up to 20 , and a bit more up to 30

# Step 1: Scale CPI and GDP columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['cpi', 'gdp']])

# Scale the forecast vector using the same scaler
scaled_new_vector = scaler.transform(forecast_vector.reshape(1, -1))

# Step 2: Compute the Euclidean distance between the forecast vector and each point in the scaled dataset
data['Distance'] = np.sqrt((scaled_features[:, 0] - scaled_new_vector[0, 0])**2 + (scaled_features[:, 1] - scaled_new_vector[0, 1])**2)

# Step 3: Sort the data by distance
data_sorted_by_distance = data.sort_values(by='Distance').reset_index(drop=True)

# Step 4: Select the top N points with the smallest distances (N is defined by num_closest_points)
closest_points = data_sorted_by_distance.head(num_closest_points)

# Step 5: Get the values of 'spx' for all data and closest points
all_spx_values = data['spx'].values
closest_spx_values = closest_points['spx'].values

# Print the closest points and their corresponding 'spx' values
print(f"Closest {num_closest_points} points:\n", closest_points)
print(f"'spx' values of the closest {num_closest_points} points:\n", closest_spx_values)

# Calculate statistics for closest points
median_closest = np.median(closest_spx_values)
mean_closest = np.mean(closest_spx_values)
std_deviation_closest = np.std(closest_spx_values)

# Calculate statistics for all data
median_all = np.median(all_spx_values)
mean_all = np.mean(all_spx_values)
std_deviation_all = np.std(all_spx_values)

# Calculate statistics for closest points
stats_closest = {
    'Median': np.median(closest_spx_values),
    'Mean': np.mean(closest_spx_values),
    'Standard Deviation': np.std(closest_spx_values),
    '25th Percentile': np.percentile(closest_spx_values, 25),
    '75th Percentile': np.percentile(closest_spx_values, 75)
}

# Calculate statistics for all data
stats_all = {
    'Median': np.median(all_spx_values),
    'Mean': np.mean(all_spx_values),
    'Standard Deviation': np.std(all_spx_values),
    '25th Percentile': np.percentile(all_spx_values, 25),
    '75th Percentile': np.percentile(all_spx_values, 75)
}
# Create a DataFrame to display the statistics
stats_df = pd.DataFrame([stats_closest, stats_all], index=[f'Closest {num_closest_points} Points', 'All Data'])
print(stats_df)

# Plot histograms
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Histogram of all 'spx' values
ax1.hist(all_spx_values, bins=10, edgecolor='black', alpha=0.7, label='All Data')
ax1.axvline(x=mean_all, color='r', linestyle='--', linewidth=1.5, label='Mean All')
ax1.axvline(x=median_all, color='g', linestyle='-.', linewidth=1.5, label='Median All')
ax1.set_title('Histogram of spx Returns')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True)

# Histogram of closest 'spx' values
ax2.hist(closest_spx_values, bins=10, edgecolor='black', alpha=0.7, label=f'Closest {num_closest_points} Points')
ax2.axvline(x=mean_closest, color='r', linestyle='--', linewidth=1.5, label='Mean Closest')
ax2.axvline(x=median_closest, color='g', linestyle='-.', linewidth=1.5, label='Median Closest')
ax2.set_title(f'Histogram of spx Returns (Closest {num_closest_points} Points)')
ax2.set_xlabel('spx Values')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
