import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

# Load datasets
train_data = pd.read_csv("../data/train_onehotEncoded20.csv")
train_data.dropna(inplace=True)

# Feature selection
selected_features = [
    # 'Aerial Duels won', 
    'Age', 
    'Assists', 
    'Attacking options created', 
    'Attempted Passes', 
    'Blocks', 
    'Clearances', 
    #'Expected Goal Contributions', 
    # 'Interceptions', 
    'Open Play Goals', 
    'Open Play Expected Goals', 
    'Percentage of Passes Completed', 
    'Progressive Passes Rec', 
    # 'Progressive Passes', 
    'Progressive Carries', 
    'Shots', 
    # 'Successful Dribbles', 
    'Touches in attacking penalty area', 
    'Tackles',
    'Value at beginning of 2022/23 season', 
    
    'Value at beginning of 2020/21 season', 
    'Value at beginning of 2021/22 season',
     

    
    # 'Country_encoded',
    
    # 'Country_Other',
    
    # 'Country_Spain',
    # 'Country_France',
    # 'Country_Germany',

    'Country_England',
    # 'Country_Italy',

    'Country_Brazil',
    
    'Country_Argentina',
    
    'Country_Portugal',
    'Country_Netherlands',
    
    'Country_Denmark',
    # 'Country_Belgium',
    # 'Country_Croatia',
    
    # 'Country_Algeria',
    # 'Country_Ghana',
    # 'Country_Austria',
    # 'Country_Nigeria',
    # 'Country_Uruguay',
    # 'Country_Morocco',
    # 'Country_Senegal',
    # 'Country_Serbia',
    # 'Country_Colombia',

    # 'Country_Norway',
    # 'Country_Scotland',
    # 'Country_Switzerland',
    # 'Country_Turkey',
    
]

X = train_data[selected_features]
y = train_data['Value at beginning of 2023/24 season']

# Create an XGBoost regressor model with regularization parameters
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,  # You can adjust this hyperparameter for regularization
    learning_rate=0.1,  # Adjust the learning rate as needed
    subsample=0.8,  # Adjust subsample to control overfitting
    colsample_bytree=0.8  # Adjust colsample_bytree to control overfitting
)

# Define a range of hyperparameters for tuning (you can adjust these values)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Use GridSearchCV to find the best hyperparameters using cross-validation
grid_search = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best hyperparameters and the corresponding XGBoost model
best_params = grid_search.best_params_
best_xgb_reg = grid_search.best_estimator_

# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(best_xgb_reg, X, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Best Hyperparameters:", best_params)
print("Cross-validated XGBoost RMSE:", cross_val_rmse)
