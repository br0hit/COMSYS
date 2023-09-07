#bm25
#-------vectorizing test and train combinedly
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize an empty list to store lines from both files
corpus = []

# Read lines from the first file and append them to the corpus list
with open('../cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())

# Read lines from the second file and append them to the corpus list
# with open('test_cleaned_text.txt', 'r') as file2:
#     corpus.extend(file2.readlines())


#bm25-------
# Initialize the TF-IDF vectorizer
from rank_bm25 import BM25Okapi
import numpy as np

tokenized_corpus = [doc.split() for doc in corpus]

# Initialize BM25Okapi with the tokenized corpus
bm25 = BM25Okapi(tokenized_corpus)

# Create a list of queries, where each query is a single document from the corpus
queries = tokenized_corpus

# Calculate BM25 scores for the queries (documents)
bm25_scores = [bm25.get_scores(query) for query in queries]

# Convert the BM25 scores into a NumPy array
bm25_matrix = np.array(bm25_scores)

# Convert the BM25 matrix to a NumPy array
bm25_ = np.array(bm25_scores)

# You can now use 'embeddings' as input features for your text classification model.

num_lines_file = len(open('../cleaned_text.txt').readlines())
print(num_lines_file)
# num_lines_file1 = len(open('test_cleaned_text.txt').readlines())
# print(num_lines_file1)


#---------------------
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(corpus)
# docs = X.toarray()
# X_train = docs[:num_lines_file][:]
# X_test = docs[num_lines_file:][:]
# print(X_train.shape)
X_train = bm25_[:num_lines_file][:]
X_test_ = bm25_[num_lines_file:][:]

from sklearn.model_selection import train_test_split


with open('../encoded_labels.txt', 'r') as file:
    y_train = np.array([int(line.strip()) for line in file])



import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Data Preparation

# Split your data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 2. Hyperparameter Tuning

# Define a range of hyperparameters to search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Create a logistic regression classifier
lr = LogisticRegression()

# Perform grid search with cross-validation
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_lr = grid_search.best_estimator_

# 3. Cross-Validation for Accuracy

# Perform k-fold cross-validation to estimate model performance
cv_scores = cross_val_score(best_lr, X_train, y_train, cv=5, scoring='accuracy')

# Calculate mean and standard deviation of cross-validation scores
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Cross-validation mean accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")

# 4. Train the Model and Evaluate on Test Data

# Fit the model on the full training data
best_lr.fit(X_train, y_train)

# Make predictions on the test set
# y_pred = best_lr.predict(x_test)

# Calculate accuracy on the test set
# test_accuracy = accuracy_score(y_test, y_pred)

# print(f"Test accuracy: {test_accuracy:.2f}")
