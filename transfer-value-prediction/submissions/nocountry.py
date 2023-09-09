import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, BayesianRidge
from xgboost import XGBRegressor

# Load datasets
train_data = pd.read_csv("../data/train_nocountry.csv")
train_data.dropna(inplace=True)
test_data = pd.read_csv("../data/test_nocountry.csv")
save_path = "predictions_stacking_countrymissing.csv"

# # Feature selection
selected_features = ['Aerial Duels won', 'Age', 'Assists', 'Attacking options created',
       'Attempted Passes', 'Blocks', 'Interceptions', 'Open Play Goals',
       'Open Play Expected Goals', 'Progressive Passes Rec',
       'Progressive Carries', 'Shots', 'Touches in attacking penalty area',
       'Tackles', 'Value at beginning of 2022/23 season',
       'Value at beginning of 2020/21 season',
       'Value at beginning of 2021/22 season']



X_train = train_data[selected_features]
y_train = train_data['Value at beginning of 2023/24 season']
X_test = test_data[selected_features]

# Transform the target variable during training
y_train_transformed = np.log1p(y_train)

# Create a list of base models
base_models = [
    # ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.9)),
    # ('Bayesian', BayesianRidge()),
    # ('ridge', Ridge(alpha=1)),
    ('lasso', Lasso(alpha=0.01)),
    # ('xgb', XGBRegressor()),  # Add XGBoost as a base model
    # ('random_forest', RandomForestRegressor())  # Add RandomForest as a base model
]

# Create a stacking ensemble with a meta-model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1))

# Filter test data for specific IDs
specific_ids = [408, 1643, 892, 453, 764, 1696, 548, 67, 1681, 772, 170, 184, 1518, 593, 522, 309, 1456]
filtered_test_data = test_data[test_data['id'].isin(specific_ids)]

# Train the stacking ensemble and prediction

stacking_model.fit(X_train, y_train)
predictions = stacking_model.predict(X_test)

# stacking_model.fit(X_train, y_train_transformed)
# predictions_transformed = stacking_model.predict(X_test)  
# predictions = np.expm1(predictions_transformed)     

# Save predictions to a new file
output_df = pd.DataFrame({'id': test_data['id'], 'label': predictions})
output_df.to_csv(save_path, index=False)
