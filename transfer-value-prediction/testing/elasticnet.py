import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, BayesianRidge

# Load datasets
train_data = pd.read_csv("../data/train_onehotEncoded20.csv")
train_data.dropna(inplace=True)

# Feature selection
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

# scaler = StandardScaler()
# X = scaler.fit_transform(X)



# Split data into training and test sets (optional if you want to use cross-validation)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Defining the models
elastic_net = ElasticNet()
bayesian_ridge = BayesianRidge()


# Define a range of alpha and l1_ratio values for hyperparameter tuning
alphas = [0.01, 0.1, 1, 10, 100]  # You can adjust this list of alphas
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # You can adjust this list of l1_ratios

param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}
grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best alpha and l1_ratio, and the corresponding Elastic Net model
best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']
best_elastic_net = grid_search.best_estimator_

# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(best_elastic_net, X, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Best Alpha for Elastic Net:", best_alpha)
print("Best l1_ratio for Elastic Net:", best_l1_ratio)
print("Cross-validated Elastic Net RMSE:", cross_val_rmse)

# No hyperparameter tuning is required for Bayesian Ridge

# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(bayesian_ridge, X, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Cross-validated Bayesian Ridge RMSE:", cross_val_rmse)