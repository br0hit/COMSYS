import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv("../data/train_onehotEncoded90.csv")
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

    # 'Country_Brazil',
    
    
]

# Define the number of runs
num_runs = 10
rmse_values = []

for _ in range(num_runs):

    X = train_data[selected_features]
    y = train_data['Value at beginning of 2023/24 season']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(1, 1000))

    # Create a Lasso regression model
    lasso = Lasso()

    # Define a range of alpha values for hyperparameter tuning
    alphas = [0.01, 0.1, 1, 10, 100]  # You can adjust this list of alphas

    # Use GridSearchCV to find the best alpha using cross-validation
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best alpha and the corresponding Lasso model
    best_alpha = grid_search.best_params_['alpha']
    best_lasso = grid_search.best_estimator_

    # Train the Lasso model with the best alpha on the entire training set
    model_final = Lasso(alpha=best_alpha)
    model_final.fit(X_train,y_train)

    # Make predictions on the test set
    lasso_predictions = model_final.predict(X_test)

    # Calculate RMSE for Lasso regression
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_predictions))

    print("Best Alpha:", best_alpha)
    print("Lasso RMSE on Test Set:", lasso_rmse)
    
    rmse_values.append(lasso_rmse)

# Calculate the mean and standard deviation of RMSE across runs
mean_rmse = np.mean(rmse_values)
std_rmse = np.std(rmse_values)

print("\n\n")
print("Mean RMSE:", mean_rmse)
print("Std RMSE:", std_rmse)