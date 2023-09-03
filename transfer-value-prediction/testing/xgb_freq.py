import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# Load datasets
train_data = pd.read_csv("../data/train_freqEncoded.csv")
# train_data = pd.read_csv("../data/train.csv")
# test_data = pd.read_csv("test.csv")

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
    'Country_encoded'

]


# Define the number of runs
num_runs = 10
rmse_values = []

for _ in range(num_runs):
    # Split data into training and testing sets with a different random seed in each run
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=np.random.randint(1, 1000))
    
    X_train = train_data[selected_features]
    y_train = train_data['Value at beginning of 2023/24 season']
    X_test = test_data[selected_features]
    y_test = test_data['Value at beginning of 2023/24 season']

    model = XGBRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    rmse_values.append(rmse)

# Calculate the mean and standard deviation of RMSE across runs
mean_rmse = np.mean(rmse_values)
std_rmse = np.std(rmse_values)

print("Mean RMSE:", mean_rmse)
print("Std RMSE:", std_rmse)

# Get feature importance scores
feature_importance = model.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Sort the DataFrame by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print or visualize feature importance scores
print(feature_importance_df)

# # Plot the feature importance scores
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title('Feature Importance Scores')
# plt.show()
