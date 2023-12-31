import numpy as np
from rank_bm25 import BM25Okapi

# Read and preprocess text data
corpus = []
with open('../../data/cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
queries = tokenized_corpus
bm25_scores = [bm25.get_scores(query) for query in queries]
bm25_matrix = np.array(bm25_scores)

# Convert the BM25 matrix to a NumPy array
bm25_ = np.array(bm25_scores)

num_lines_file = len(open('../../data/cleaned_text.txt').readlines())

X_train = bm25_[:num_lines_file][:]
X_test = bm25_[num_lines_file:][:]

file_name = 'bm25_vect.npz'

# Save the arrays into the .npz file
np.savez(file_name, X_train=X_train,X_test=X_test)




import numpy as np
from rank_bm25 import BM25Okapi

# Read and preprocess text data
corpus = []
with open('../../data/cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())
with open('../../data/test_cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
queries = tokenized_corpus
bm25_scores = [bm25.get_scores(query) for query in queries]
bm25_matrix = np.array(bm25_scores)

# Convert the BM25 matrix to a NumPy array
bm25_ = np.array(bm25_scores)

num_lines_file = len(open('../../data/cleaned_text.txt').readlines())

X_train = bm25_[:num_lines_file][:]
X_test = bm25_[num_lines_file:][:]

# Load custom features from a CSV file (replace 'custom_features.csv' with your actual file path)
custom_features = np.genfromtxt('../../data/custom_features.csv', delimiter=',', skip_header=1)
test_custom_features = np.genfromtxt('../../data/test_custom_features.csv', delimiter=',', skip_header=1)

# Stack the custom features horizontally with the BM25 vectors
X_train_with_custom = np.hstack((X_train, custom_features))
X_test_with_custom = np.hstack((X_test, test_custom_features))

file_name = 'bm25_vect_with_custom1.npz'

# Save the arrays into the .npz file
np.savez(file_name, X_train=X_train_with_custom,X_test=X_test_with_custom)
