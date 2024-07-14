import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load data from Excel file
data = pd.read_excel("df.xlsx")

# Separate features and target variable
X = data[['cpi', 'gdp']]
y = data['spx']

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

# Define the parameter grid
param_grid = {
    'knn__n_neighbors': [10, 15,16,17,18,19, 20,21,22,23,24, 25, 30,35 ,40,45,50]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X, y)

# Get the best parameter and the corresponding score
best_n_neighbors = grid_search.best_params_['knn__n_neighbors']
best_score = -grid_search.best_score_

print(f"Best number of neighbors: {best_n_neighbors}")
print(f"Best score (neg_mean_squared_error): {best_score}")

# Extract results for plotting
results = grid_search.cv_results_
mean_scores = -results['mean_test_score']  # Convert scores to positive values
param_n_neighbors = results['param_knn__n_neighbors']

# Plot number of neighbors vs average cross-validation error
plt.figure(figsize=(10, 6))
plt.plot(param_n_neighbors, mean_scores, marker='o')
plt.title('Number of Neighbors vs Average Cross-Validation Error')
plt.xlabel('Number of Neighbors')
plt.ylabel('Average Cross-Validation Error (Negative MSE)')
plt.grid(True)
plt.show()

# Use the best number of neighbors to predict the new data point
best_model = grid_search.best_estimator_
