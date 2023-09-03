import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load datasets
train_data = pd.read_csv("../data/train.csv")
train_data.dropna(inplace=True)
test_data = pd.read_csv("../data/test.csv")
test_data.dropna(inplace=True)
save_path = "predictions_ridge.csv"

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

X_train = train_data[selected_features]
y_train = train_data['Value at beginning of 2023/24 season']
X_test = test_data[selected_features]

# Model selection and training
ridge_alpha = 1  # Alpha value for Ridge Regression
model = Ridge(alpha=ridge_alpha)
model.fit(X_train, y_train)

# Model prediction
predictions = model.predict(X_test)

# Save predictions to a new file
output_df = pd.DataFrame({'id': test_data['id'], 'label': predictions})
output_df.to_csv(save_path, index=False)
