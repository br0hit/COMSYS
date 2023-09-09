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
