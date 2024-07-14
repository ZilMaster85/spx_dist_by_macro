import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data from Excel file
data = pd.read_excel("df.xlsx")

# Step 1: Scale CPI and GDP columns
scaler = StandardScaler()
data[['cpi_scaled', 'gdp_scaled']] = scaler.fit_transform(data[['cpi', 'gdp']])

# Define the conditions for each group
conditions = {
    "high gdp high cpi": (data['cpi_scaled'] > 0.75) & (data['gdp_scaled'] > 0.75),
    "high gdp low inf": (data['cpi_scaled'] < -0.75) & (data['gdp_scaled'] > 0.75),
    "low gdp high inf": (data['cpi_scaled'] > 0.75) & (data['gdp_scaled'] < -0.75),
    "low gdp low inf": (data['cpi_scaled'] < -0.75) & (data['gdp_scaled'] < -0.75),
}

# Initialize a dictionary to store statistics for each group
stats_dict = {}

# Loop through each group to calculate statistics and plot histograms
for group_name, condition in conditions.items():
    group_data = data[condition]
    spx_values = group_data['spx'].values

    # Calculate statistics for the group
    stats = {
        "Count": len(spx_values),
        'Median': np.median(spx_values),
        'Mean': np.mean(spx_values),
        'Standard Deviation': np.std(spx_values),
        '25th Percentile': np.percentile(spx_values, 25),
        '75th Percentile': np.percentile(spx_values, 75)
    }
    stats_dict[group_name] = stats

    # Plot histogram for the group
    plt.figure(figsize=(10, 6))
    plt.hist(spx_values, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=stats['Mean'], color='r', linestyle='--', linewidth=1.5, label='Mean')
    plt.axvline(x=stats['Median'], color='g', linestyle='-.', linewidth=1.5, label='Median')
    plt.title(f'Histogram of spx Returns ({group_name})')
    plt.xlabel('spx Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create a DataFrame to display the statistics for all groups
stats_df = pd.DataFrame(stats_dict).T
print(stats_df)
stats_df.to_excel("group_statistics.xlsx", index=True)
