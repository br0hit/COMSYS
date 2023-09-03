import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv("../data/train.csv")
train_data.dropna(inplace=True)

# Feature selection
selected_features = [
    'Aerial Duels won', 
    'Age', 
    'Assists', 
    'Attacking options created', 
    'Attempted Passes', 
    'Blocks', 
    'Clearances', 
    'Expected Goal Contributions', 
    'Interceptions', 
    'Open Play Goals', 
    'Open Play Expected Goals', 
    'Percentage of Passes Completed', 
    'Progressive Passes Rec', 
    'Progressive Passes', 
    'Progressive Carries', 
    'Shots', 
    'Successful Dribbles', 
    'Touches in attacking penalty area', 
    'Tackles', 
    'Value at beginning of 2020/21 season', 
    'Value at beginning of 2021/22 season', 
    'Value at beginning of 2022/23 season',
]

X = train_data[selected_features]
y = train_data['Value at beginning of 2023/24 season']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor(random_state=42)

# Define a range of hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Use GridSearchCV to find the best hyperparameters using cross-validation
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding Random Forest Regressor model
best_params = grid_search.best_params_
best_rf_regressor = grid_search.best_estimator_

# Train the Random Forest Regressor with the best hyperparameters on the entire training set
best_rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = best_rf_regressor.predict(X_test)

# Calculate RMSE for Random Forest Regressor
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

print("Best Hyperparameters:", best_params)
print("Random Forest RMSE on Test Set:", rf_rmse)
