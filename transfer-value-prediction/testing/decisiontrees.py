import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

# Load datasets
train_data = pd.read_csv("../data/train_onehotEncoded20.csv")
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
    'Value at beginning of 2022/23 season',
    
    'Value at beginning of 2020/21 season', 
    'Value at beginning of 2021/22 season', 

    
    # 'Country_encoded',
    
    'Country_Other',
    
    'Country_Spain',
    'Country_France',
    'Country_Germany',
    
    'Country_England',
    'Country_Italy',

    'Country_Brazil',
    
    'Country_Argentina',
    
    # 'Country_Portugal',
    # 'Country_Netherlands',    
]

X = train_data[selected_features]
y = train_data['Value at beginning of 2023/24 season']

# Create a Random Forest regressor model with regularization parameters
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=45)

# Define a range of hyperparameters for tuning (you can adjust these values)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best hyperparameters using cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best hyperparameters and the corresponding Random Forest model
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(best_rf, X, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Best Hyperparameters:", best_params)
print("Cross-validated Random Forest RMSE:", cross_val_rmse)
