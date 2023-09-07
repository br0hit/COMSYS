import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, BayesianRidge
from xgboost import XGBRegressor
import xgboost as xgb


# Load datasets
train_data = pd.read_csv("../data/train_onehotEncoded20.csv")
train_data.dropna(inplace=True)

# Initialize RFE with the Ridge model and the number of features to select
num_features_to_select = 19  # You can adjust this number


# Create a list of base models
base_models = [
    ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.9)),
    ('Bayesian', BayesianRidge()),
    ('ridge', Ridge(alpha=1)),
    ('lasso', Lasso(alpha=0.01)),
    ('xgb', xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,  # You can adjust this hyperparameter for regularization
    learning_rate=0.1,  # Adjust the learning rate as needed
    subsample=0.7,  # Adjust subsample to control overfitting
    colsample_bytree=0.8  # Adjust colsample_bytree to control overfitting
)),  # Add XGBoost as a base model
]


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
    
    # 'Country_encoded'
    
    
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

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Transform the target variable during training
# y_train_transformed = np.log1p(y_train)

# Create a stacking ensemble with a meta-model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1))

# Train the stacking ensemble and prediction

# stacking_model.fit(X_train, y_train)
# predictions = stacking_model.predict(X_test)

# stacking_model.fit(X_train, y_train_transformed)
# predictions_transformed = stacking_model.predict(X_test)
# predictions = np.expm1(predictions_transformed)     


# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(stacking_model, X, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Stacked RMSE on Test Set:", cross_val_rmse)
print("Models used : ",base_models)

# rfe = RFE(estimator=stacking_model, n_features_to_select= num_features_to_select)

# # Fit RFE to your data
# rfe.fit(X, y)

# # Get the selected features and their rankings
# selected_features = X.columns[rfe.support_]
# feature_rankings = rfe.ranking_

# # Print selected features and their rankings
# print("Selected Features:", selected_features)
# print("Number of features used:", num_features_to_select)


# # Use the selected features for your data
# X_selected = X[selected_features]

# # Create a Ridge regression model with the best alpha
# model_with_selected_features = stacking_model

# # Perform cross-validation to calculate RMSE using the selected features
# cross_val_rmse_selected = np.sqrt(-cross_val_score(model_with_selected_features, X_selected, y, cv=5, scoring='neg_mean_squared_error').mean())

# print("Cross-validated Selection RMSE with selected features:", cross_val_rmse_selected)