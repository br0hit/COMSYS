import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.ensemble import StackingClassifier

SEED=123

file_name = '../preprocessing/vectorization/bm25_vect_with_custom1.npz'

# Load the .npz file
loaded_data = np.load(file_name)

# Access and load individual arrays from the loaded data
X_train = loaded_data['X_train']

with open('../data/encoded_labels.txt', 'r') as file:
    y_train = np.array([int(line.strip()) for line in file])


# Create a list of base models
base_models = [
    ('RandomForest', RandomForestClassifier(random_state=SEED)),
    ('Logistic', LogisticRegression(random_state=SEED)),
    ('GradientBoost', GradientBoostingClassifier(random_state=SEED)),
]

# Create a stacking ensemble with a meta-model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(random_state=SEED, n_estimators=100))


