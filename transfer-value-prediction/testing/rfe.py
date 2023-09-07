import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load datasets
train_data = pd.read_csv("../data/train_nocountry.csv")
train_data.dropna(inplace=True)

best_alpha = 1

# Specify the number of features you want to select
n_features_to_select = 21     # You can adjust this number



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


# # Standardize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(train_data[selected_features])
# y = train_data['Value at beginning of 2023/24 season']

X = train_data[selected_features]
y = train_data['Value at beginning of 2023/24 season']

# Create a Ridge regression model
ridge = Ridge()


# Create an RFE object
rfe = RFE(estimator=ridge, n_features_to_select=n_features_to_select)

# Fit the RFE to your data
rfe.fit(X, y)

# Get the best set of features
best_features = train_data[selected_features].columns[rfe.support_]

# Print the results
print("Best Features:", best_features.tolist())

# Fit the Ridge model to the selected features
X_best = X[:, rfe.support_]
best_ridge = Ridge(alpha=best_alpha)
best_ridge.fit(X_best, y)

# Perform cross-validation to calculate RMSE
cross_val_rmse = np.sqrt(-cross_val_score(best_ridge, X_best, y, cv=5, scoring='neg_mean_squared_error').mean())

print("Cross-validated Ridge RMSE on Best Features:", cross_val_rmse)
