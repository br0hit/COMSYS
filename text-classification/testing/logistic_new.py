import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

SEED = 123
save_path = 'output_file_final_logistic.csv'

file_name = '../preprocessing/vectorization/bm25_vect_with_custom1.npz'

# Load the .npz file
loaded_data = np.load(file_name)

# Access and load individual arrays from the loaded data
X = loaded_data['X_train']

with open('../data/encoded_labels.txt', 'r') as file:
    y = np.array([int(line.strip()) for line in file])

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Define hyperparameter grid for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Penalty term
    'solver': ['liblinear', 'lbfgs', 'newton-cg'],  # Solver algorithm
    'multi_class': ['ovr', 'multinomial'],  # Multi-class strategy
    'class_weight': [None, 'balanced'],  # Class weights
    'dual': [False],  # Solve the dual problem (recommended)
    'max_iter': [100, 200, 300]  # Maximum number of iterations
}

# Create a Logistic Regression model
logistic_model = LogisticRegression(random_state=SEED)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model and hyperparameters
best_logistic_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Train the best model on the full training set
best_logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = best_logistic_model.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy on Test Set: {accuracy}")
