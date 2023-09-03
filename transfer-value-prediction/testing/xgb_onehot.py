import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    'Value at beginning of 2020/21 season', 
    'Value at beginning of 2021/22 season', 
    'Value at beginning of 2022/23 season',
    
    # 'Country_encoded',
    
    'Country_Other',
    
    'Country_Spain',
    'Country_France',
    'Country_Germany',

    'Country_England',
    'Country_Italy',

    'Country_Brazil',
    
    'Country_Argentina',
    
    'Country_Portugal',
    'Country_Netherlands',
    
    'Country_Denmark',
    'Country_Belgium',
    'Country_Croatia',
    
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

# Optional: You can scale the features as XGBoost is sensitive to feature scales
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Create an XGBoost regression model
xgb_model = xgb.XGBRegressor()

# Define a range of hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 500, 1000],  # Number of trees
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
    'max_depth': [3, 4, 5],  # Maximum depth of a tree
    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight (hessian) needed in a child
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting the trees
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for fitting the trees
}

# Use GridSearchCV to find the best hyperparameters using cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best hyperparameters and the corresponding XGBoost model
best_params = grid_search.best_params_
best_xgb_model = grid_search.best_estimator_

# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(best_xgb_model, X, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Best Hyperparameters:", best_params)
print("Cross-validated XGBoost RMSE:", cross_val_rmse)
