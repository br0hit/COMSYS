import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

file_name = '../preprocessing/vectorization/bm25_vect.npz'

# Load the .npz file
loaded_data = np.load(file_name)

# Access and load individual arrays from the loaded data
X_train = loaded_data['X_train']
X_test = loaded_data['X_test']

# Load labels
with open('../data/encoded_labels.txt', 'r') as file:
    y_train = np.array([int(line.strip()) for line in file])

# Define hyperparameter grid for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],  # You can customize the max depth values
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create a RandomForest classifier
rf = RandomForestClassifier()

# Hyperparameter tuning with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Cross-validation for classification metrics
cv_accuracy_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
cv_precision_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='precision_macro')
cv_recall_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='recall_macro')
cv_f1_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1_macro')

mean_accuracy = np.mean(cv_accuracy_scores)
mean_precision = np.mean(cv_precision_scores)
mean_recall = np.mean(cv_recall_scores)
mean_f1 = np.mean(cv_f1_scores)

std_accuracy = np.std(cv_accuracy_scores)
std_precision = np.std(cv_precision_scores)
std_recall = np.std(cv_recall_scores)
std_f1 = np.std(cv_f1_scores)

print(f"Cross-validation mean accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
print(f"Cross-validation mean precision: {mean_precision:.2f} ± {std_precision:.2f}")
print(f"Cross-validation mean recall: {mean_recall:.2f} ± {std_recall:.2f}")
print(f"Cross-validation mean F1-score: {mean_f1:.2f} ± {std_f1:.2f}")
