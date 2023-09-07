# Import necessary libraries
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.metrics import accuracy_score
from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split

# Read and preprocess text data
corpus = []
with open('../data/cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
queries = tokenized_corpus
bm25_scores = [bm25.get_scores(query) for query in queries]
bm25_matrix = np.array(bm25_scores)

# Convert the BM25 matrix to a NumPy array
bm25_ = np.array(bm25_scores)

num_lines_file = len(open('../data/cleaned_text.txt').readlines())
print(num_lines_file)

X_train = bm25_[:num_lines_file][:]
X_test_ = bm25_[num_lines_file:][:]

# Load labels
with open('../data/encoded_labels.txt', 'r') as file:
    y_train = np.array([int(line.strip()) for line in file])

# Define hyperparameter grid for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300]
}

# Create an XGBoost classifier
xgb = XGBClassifier()

# Hyperparameter tuning with cross-validation
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

# Cross-validation for accuracy
cv_scores = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='accuracy')
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Cross-validation mean accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
