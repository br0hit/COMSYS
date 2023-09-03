import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, BayesianRidge

# Load datasets
train_data = pd.read_csv("../data/train_onehotEncoded20.csv")
train_data.dropna(inplace=True)
test_data = pd.read_csv("../data/test_full_onehotEncoded20.csv")
save_path = "predictions_ensemble.csv"

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

X_train = train_data[selected_features]
y_train = train_data['Value at beginning of 2023/24 season']
X_test = test_data[selected_features]

# Create a list of models for ensemble learning
models = [
    ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.9)),
    ('Bayesian', BayesianRidge()),
    ('ridge', Ridge(alpha=1)),  # Ridge Regressor model
    ('lasso', Lasso(alpha=0.01))
]

# Create an ensemble of models
ensemble_model = VotingRegressor(models)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Model prediction
predictions = ensemble_model.predict(X_test)

# Save predictions to a new file
output_df = pd.DataFrame({'id': test_data['id'], 'label': predictions})
output_df.to_csv(save_path, index=False)
