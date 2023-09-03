import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb  # Import XGBoost
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

# Create an XGBoost Regressor model
xgb_regressor = xgb.XGBRegressor(random_state=42)

# Define a range of hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'max_depth': [3, 4, 5],  # Maximum depth of trees
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
    'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight (hessian) needed in a child
    'subsample': [0.8, 0.9, 1.0],  # Subsample ratio of the training data
    'colsample_bytree': [0.8, 0.9, 1.0],  # Subsample ratio of columns when constructing each tree
}

# Use GridSearchCV to find the best hyperparameters using cross-validation
grid_search = GridSearchCV(xgb_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding XGBoost Regressor model
best_params = grid_search.best_params_
best_xgb_regressor = grid_search.best_estimator_

# Train the XGBoost Regressor with the best hyperparameters on the entire training set
best_xgb_regressor.fit(X_train, y_train)

# Make predictions on the test set
xgb_predictions = best_xgb_regressor.predict(X_test)

# Calculate RMSE for XGBoost Regressor
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

print("Best Hyperparameters:", best_params)
print("XGBoost RMSE on Test Set:", xgb_rmse)
