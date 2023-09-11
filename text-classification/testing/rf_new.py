import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

SEED = 123
save_path = 'output_file_final_random_forest.csv'

file_name = '../preprocessing/vectorization/bm25_vect_with_custom1.npz'

# Load the .npz file
loaded_data = np.load(file_name)

# Access and load individual arrays from the loaded data
X = loaded_data['X_train']

with open('../data/encoded_labels.txt', 'r') as file:
    y = np.array([int(line.strip()) for line in file])

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Define hyperparameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False],  # Whether bootstrap samples are used
    'class_weight': [None, 'balanced']  # Class weights
}

# Create a RandomForestClassifier model
rf_model = RandomForestClassifier(random_state=SEED)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model and hyperparameters
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Train the best model on the full training set
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = best_rf_model.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy on Test Set: {accuracy}")
