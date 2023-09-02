import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest
from sklearn.metrics import mean_squared_error

# Load datasets
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

save_path = "rf.csv"


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
    'Value at beginning of 2022/23 season'
]

# # Convert 'Country' column to one-hot encoding
# train_data = pd.get_dummies(train_data, columns=['Country'], drop_first=True)
# test_data = pd.get_dummies(test_data, columns=['Country'], drop_first=True)

# X_train = train_data[selected_features + train_data.columns.tolist()]
# y_train = train_data['Value at beginning of 2023/24 season']
# X_test = test_data[selected_features + test_data.columns.tolist()]

X_train = train_data[selected_features ]
y_train = train_data['Value at beginning of 2023/24 season']
X_test = test_data[selected_features]

# Model selection and training
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest model with 100 trees
model.fit(X_train, y_train)

# Model prediction
predictions = model.predict(X_test)

# Save predictions to a new file
output_df = pd.DataFrame({'id': test_data['id'], 'label': predictions})
output_df.to_csv(save_path, index=False)
