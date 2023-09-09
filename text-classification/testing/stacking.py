import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import StackingClassifier

SEED = 123

file_name = '../preprocessing/vectorization/bm25_vect_with_custom1.npz'

# Load the .npz file
loaded_data = np.load(file_name)

# Access and load individual arrays from the loaded data
X = loaded_data['X_train']

with open('../data/encoded_labels.txt', 'r') as file:
    y = np.array([int(line.strip()) for line in file])

# Create a list of base models
base_models = [
    ('RandomForest', RandomForestClassifier(random_state=SEED)),
    ('Logistic', LogisticRegression(random_state=SEED)),
    ('GradientBoost', GradientBoostingClassifier(random_state=SEED)),
]

# Create a stacking ensemble with a meta-model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(random_state=SEED, n_estimators=100))

# Use cross-validation to estimate the accuracy of the stacking model
cv_scores = cross_val_score(stacking_model, X, y, cv=5)  # You can adjust the number of folds (cv) as needed

# Calculate the mean accuracy and standard deviation of the cross-validation scores
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Mean Accuracy: {mean_accuracy}")
print(f"Standard Deviation of Accuracy: {std_accuracy}")
