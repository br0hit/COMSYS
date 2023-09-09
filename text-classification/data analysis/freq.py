import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Sample data (replace this with your actual data)

# Load text data
with open('../data/train_text.txt', 'r', encoding='utf-8') as file:
    x_train = file.readlines()

# Load labels
with open('../data/encoded_labels.txt', 'r') as file:
    y_train = [int(line.strip()) for line in file]

# Tokenize and count words for each label
label_word_counts = {}
for label in set(y_train):
    label_indices = [i for i, y in enumerate(y_train) if y == label]
    label_tweets = [x_train[i] for i in label_indices]
    
    # Tokenize and count word frequencies in label_tweets
    words = [word.lower() for tweet in label_tweets for word in word_tokenize(tweet)]
    word_freq = Counter(words)
    
    label_word_counts[label] = word_freq

# Combine word frequencies across all labels
total_word_freq = Counter()
for word_freq in label_word_counts.values():
    total_word_freq += word_freq

# Get the top 100 high-frequency words in decreasing order of total frequency
top_words = [word for word, _ in total_word_freq.most_common(100)]

# Create a DataFrame to store the word frequencies
df = pd.DataFrame()

# Populate the DataFrame with word frequencies for each label
df['Word'] = top_words
df['Total_Frequency'] = [total_word_freq[word] for word in top_words]

for label in set(y_train):
    df[label] = [label_word_counts[label][word] for word in top_words]

# Export the DataFrame to a CSV file
df.to_csv('word_frequencies.csv', index=False)
