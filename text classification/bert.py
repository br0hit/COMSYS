from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

# Initialize SBERT model
sentence_transformer = SentenceTransformer("bert-base-nli-mean-tokens")

# Input and output file paths
input_file_path = "cleaned_text.txt"
output_file_path = "sentence_embeddings.npy"

# Create a function to get sentence embeddings
def get_sentence_embeddings(sentences):
    sentence_embeddings = sentence_transformer.encode(sentences, convert_to_numpy=True)
    return sentence_embeddings

# Read sentences from the input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file]

# Get sentence embeddings
sentence_embeddings = get_sentence_embeddings(sentences)

# Save the sentence embeddings as a .npy file
np.save(output_file_path, sentence_embeddings)

print(f"Sentence embeddings saved to {output_file_path}")
