from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import csv
# Sample data
important_words = {"class1": ['anger', 'emotions', 'emotional', 'rage', 'tired', 'intense'], "class2": ['anxiety', 'worry', 'mind', 'thoughts', 'uncertainty', 'worries', 'always'],"class3":[ 'sadness', 'heavy', 'pain', 'emotions', 'anger', 'even'],"class4":['others', 'success', 'expect', 'attention', 'people', 'achievements', 'greatness', 'exceptional', 'often', 'entitled'],"class5":['panic', 'fear', 'attack', 'body', 'during', 'terrified', 'losing']}  # Important words for each class

corpus = []
with open('../data/cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())

# Create binary features based on important words
def create_custom_features(text, important_words):
    features = []
    for words in important_words.values():
        feature = [1 if word in text else 0 for word in words]
        features.extend(feature)
    return features

x_custom_features = [create_custom_features(text, important_words) for text in corpus]

# Specify the path where you want to save the custom features as a CSV file
custom_features_csv_path = "../data/custom_features.csv"

# Save custom features as a CSV file
with open(custom_features_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header row with feature names (important word names)
    header = []
    for words in important_words.values():
        header.extend(words)
    csv_writer.writerow(header)
    
    # Write binary feature vectors for each text sample
    for feature_vector in x_custom_features:
        csv_writer.writerow(feature_vector)

print(f"Custom features saved to {custom_features_csv_path}.")


corpus = []
with open('../data/test_cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())

# Create binary features based on important words
def create_custom_features(text, important_words):
    features = []
    for words in important_words.values():
        feature = [1 if word in text else 0 for word in words]
        features.extend(feature)
    return features

x_custom_features = [create_custom_features(text, important_words) for text in corpus]

# Specify the path where you want to save the custom features as a CSV file
custom_features_csv_path = "../data/test_custom_features.csv"

# Save custom features as a CSV file
with open(custom_features_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header row with feature names (important word names)
    header = []
    for words in important_words.values():
        header.extend(words)
    csv_writer.writerow(header)
    
    # Write binary feature vectors for each text sample
    for feature_vector in x_custom_features:
        csv_writer.writerow(feature_vector)

print(f"Custom features saved to {custom_features_csv_path}.")
