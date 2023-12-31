import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV


# Load datasets
train_data = pd.read_csv("../data/train_nocountry.csv")
train_data.dropna(inplace=True)

# Initialize RFE with the Ridge model and the number of features to select
num_features_to_select = 16  # You can adjust this number

# Feature selection
# Feature selection
selected_features = [
    # 'Aerial Duels won', 
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
    
    # 'Country_Other',
    
    # 'Country_Spain',
    # 'Country_France',
    # 'Country_Germany',

    # 'Country_England',
    # 'Country_Italy',

    # 'Country_Brazil',
    
    # 'Country_Argentina',
    
    # 'Country_Portugal',
    # 'Country_Netherlands',
    
    # 'Country_Denmark',
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

# # Calculate y_log_exp
# y = np.exp(np.log(y))

# Split data into training and test sets (optional if you want to use cross-validation)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Create a Lasso regression model
lasso = Lasso()

# Define a range of alpha values for hyperparameter tuning
alphas = [0.01, 0.1, 1, 10, 100]  # You can adjust this list of alphas

# Use GridSearchCV to find the best alpha using cross-validation
param_grid = {'alpha': alphas}
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best alpha and the corresponding Lasso model
best_alpha = grid_search.best_params_['alpha']
best_lasso = grid_search.best_estimator_

# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(best_lasso, X, y , cv=5, scoring='neg_mean_squared_error').mean())

print("Best Alpha:", best_alpha)
print("Cross-validated Lasso RMSE:", cross_val_rmse)

# Create a lasso regression model with the best alpha
best_lasso = Lasso(alpha=best_alpha)


rfe = RFE(estimator=best_lasso, n_features_to_select= num_features_to_select)

# Fit RFE to your data
rfe.fit(X, y)

# Get the selected features and their rankings
selected_features = X.columns[rfe.support_]
feature_rankings = rfe.ranking_

# Print selected features and their rankings
print("Selected Features:", selected_features)
print("Number of features used:", num_features_to_select)


# Use the selected features for your data
X_selected = X[selected_features]

# Create a lasso regression model with the best alpha
lasso_with_selected_features = Lasso(alpha=best_alpha)

# Perform cross-validation to calculate RMSE using the selected features
cross_val_rmse_selected = np.sqrt(-cross_val_score(lasso_with_selected_features, X_selected, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Cross-validated lasso RMSE with selected features:", cross_val_rmse_selected)
