# Load text data
with open('../data/train_text.txt', 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Load labels
with open('../data/encoded_labels.txt', 'r') as file:
    labels = [int(line.strip()) for line in file]

total_length = sum(len(tweet.split()) for tweet in tweets)
average_length = total_length / len(tweets)
print(f"Average Tweet Length: {average_length:.2f} words")


from collections import Counter
import matplotlib.pyplot as plt

# Tokenize tweets into words
words = [word for tweet in tweets for word in tweet.split()]

# Calculate word frequencies
word_freq = Counter(words)

# Plot the distribution of word frequencies
plt.figure(figsize=(10, 6))
plt.hist(word_freq.values(), bins=range(1, 21), align='left', edgecolor='k')
plt.xlabel('Word Frequency')
plt.ylabel('Number of Words')
plt.title('Distribution of Word Frequencies')
plt.show()

# Find the most common words
most_common_words = word_freq.most_common(100)
print("Most Common Words:")
for word, freq in most_common_words:
    print(f"{word}: {freq} times")


import pandas as pd
import seaborn as sns

# Create a DataFrame with tweets and labels
data = pd.DataFrame({'Tweet': tweets, 'Label': labels})

# Count the number of tweets per class
class_counts = data['Label'].value_counts()

# Plot the distribution of tweets across classes
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.xlabel('Class')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Tweets Across Classes')
plt.show()

# Print the class distribution
print("Class Distribution:")
print(class_counts)


