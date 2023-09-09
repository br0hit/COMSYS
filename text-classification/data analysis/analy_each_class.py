
# Load text data
with open('../data/train_text.txt', 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Load labels
with open('../data/encoded_labels.txt', 'r') as file:
    labels = [int(line.strip()) for line in file]


# Create a dictionary to store tweets by class
tweets_by_class = {}
for label in set(labels):
    class_tweets = [tweet for tweet, tweet_label in zip(tweets, labels) if tweet_label == label]
    tweets_by_class[label] = class_tweets


from collections import Counter

# Define a function to find most common words
def most_common_words(class_tweets, top_n=10):
    words = [word for tweet in class_tweets for word in tweet.split()]
    word_freq = Counter(words)
    common_words = word_freq.most_common(top_n)
    return common_words

# Analyze each class
for label, class_tweets in tweets_by_class.items():
    print(f"Class {label}:")
    common_words = most_common_words(class_tweets)
    for word, freq in common_words:
        print(f"{word}: {freq} times")
    print("\n")


##### specific imp words for that class
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)  # Assuming 'tweets' contains your text data

# Create a separate Random Forest classifier for each class
classifiers = {}
for label in set(labels):
    # Create a binary label vector for the current class
    binary_labels = [1 if l == label else 0 for l in labels]
    
    # Train a Random Forest classifier for the current class
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X, binary_labels)
    
    # Store the classifier
    classifiers[label] = rf_classifier

# Get feature importance scores for each class
word_importance_per_class = {}
for label, classifier in classifiers.items():
    feature_importance = classifier.feature_importances_
    word_importance_per_class[label] = dict(zip(vectorizer.get_feature_names_out(), feature_importance))

# Sort words by importance for each class
for label, word_importance in word_importance_per_class.items():
    sorted_class_words = sorted(word_importance, key=lambda x: word_importance[x], reverse=True)
    print(f"Class {label} Important Words:")
    print(sorted_class_words[:10])  # Print the top 10 important words for each class


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(tfidf_matrix, labels)

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get feature log probabilities for each class
feature_log_probs = model.feature_log_prob_

# Identify top words for each class
top_words_per_class = {}
for i, class_name in enumerate(model.classes_):
    top_word_indices = feature_log_probs[i].argsort()[::-1][:10]  # Top 10 words
    top_words = [feature_names[idx] for idx in top_word_indices]
    top_words_per_class[class_name] = top_words

# Print top words for each class
for label, top_words in top_words_per_class.items():
    print(f"Class {label} Important Words:")
    print(top_words)



#---unique to that class
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)  # Assuming 'tweets' contains your text data

# Create a separate Random Forest classifier for each class
classifiers = {}
for label in set(labels):
    # Create a binary label vector for the current class
    binary_labels = [1 if l == label else 0 for l in labels]
    
    # Train a Random Forest classifier for the current class
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X, binary_labels)
    
    # Store the classifier
    classifiers[label] = rf_classifier

# Get feature importance scores for each class
word_importance_per_class = {}
for label, classifier in classifiers.items():
    feature_importance = classifier.feature_importances_
    word_importance_per_class[label] = dict(zip(vectorizer.get_feature_names_out(), feature_importance))

# Calculate the set of common words that appear in all sets of important words
common_words = set.intersection(*[set(words.keys()) for words in word_importance_per_class.values()])

# Sort words by importance for each class and exclude common words
for label, word_importance in word_importance_per_class.items():
    sorted_class_words = sorted(word_importance, key=lambda x: word_importance[x], reverse=True)
    
    # Exclude common words
    unique_class_words = [word for word in sorted_class_words if word not in common_words]
    
    print(f"Class {label} Important Words (Excluding Common Words):")
    print(unique_class_words[:10])  # Print the top 10 important words for each class (excluding common words)




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(tfidf_matrix, labels)

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get feature log probabilities for each class
feature_log_probs = model.feature_log_prob_

# Calculate the set of words that are common to all classes
common_words = set.intersection(*[set(feature_names) for _ in model.classes_])

# Identify top words for each class excluding common words
top_words_per_class = {}
for i, class_name in enumerate(model.classes_):
    class_feature_log_prob = feature_log_probs[i]
    
    # Filter out common words
    filtered_indices = [idx for idx, word in enumerate(feature_names) if word not in common_words]
    
    # Get the top 10 words for this class (excluding common words)
    top_word_indices = sorted(filtered_indices, key=lambda idx: class_feature_log_prob[idx], reverse=True)[:10]
    top_words = [feature_names[idx] for idx in top_word_indices]
    
    top_words_per_class[class_name] = top_words

# Print top words for each class
for label, top_words in top_words_per_class.items():
    print(f"Class {label} Important Words (Excluding Common Words):")
    print(top_words)
