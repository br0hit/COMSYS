

import nltk
nltk.download('punkt')

# Define a function to remove stopwords from a text file
def remove_stopwords(input_file, output_file, stopwords_file):
    # Read the stopwords from the stopwords file
    with open(stopwords_file, 'r', encoding='utf-8') as stopwords_file:
        stopwords = set(stopwords_file.read().split())

    # Initialize an empty list to store the filtered sentences
    filtered_sentences = []

    # Read the input text file
    with open(input_file, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            # Tokenize the sentence into words
            words = nltk.word_tokenize(line)

            # Remove stopwords from the sentence
            filtered_words = [word for word in words if word.lower() not in stopwords]

            # Join the filtered words back into a sentence
            filtered_sentence = ' '.join(filtered_words)

            # Append the filtered sentence to the list
            filtered_sentences.append(filtered_sentence)

    # Write the filtered sentences to the output file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(filtered_sentences))

# Usage example:
input_file = '../data/train_text.txt'  # Replace with your input file path
output_file = '../data/cleaned_text.txt'  # Replace with the desired output file path
stopwords_file = 'stopwords.txt'  # Replace with your stopwords file path

remove_stopwords(input_file, output_file, stopwords_file)


input_file_path = output_file

# Specify the path where you want to save the lowercased text
output_file_path = output_file

# Open the input file for reading
with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Read the content of the file
    text = input_file.read()

# Convert the text to lowercase
lowercased_text = text.lower()

# Open the output file for writing
with open(output_file_path, "w", encoding="utf-8") as output_file:
    # Write the lowercased text to the output file
    output_file.write(lowercased_text)





import string

# Specify the path to your input text file
input_file_path = output_file_path

# Specify the path where you want to save the text without punctuation
output_file_path = output_file_path

# Function to remove punctuation from text
def remove_punctuation(text):
    # Create a translation table to remove punctuation characters
    translator = str.maketrans('', '', string.punctuation)
    
    # Use translate to remove punctuation
    text_without_punctuation = text.translate(translator)
    
    return text_without_punctuation

# Open the input file for reading
with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Read the content of the file
    text = input_file.read()

# Remove punctuation from the text
text_without_punctuation = remove_punctuation(text)

# Open the output file for writing
with open(output_file_path, "w", encoding="utf-8") as output_file:
    # Write the text without punctuation to the output file
    output_file.write(text_without_punctuation)







import nltk
nltk.download('punkt')

# Define a function to remove stopwords from a text file
def remove_stopwords(input_file, output_file, stopwords_file):
    # Read the stopwords from the stopwords file
    with open(stopwords_file, 'r', encoding='utf-8') as stopwords_file:
        stopwords = set(stopwords_file.read().split())

    # Initialize an empty list to store the filtered sentences
    filtered_sentences = []

    # Read the input text file
    with open(input_file, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            # Tokenize the sentence into words
            words = nltk.word_tokenize(line)

            # Remove stopwords from the sentence
            filtered_words = [word for word in words if word.lower() not in stopwords]

            # Join the filtered words back into a sentence
            filtered_sentence = ' '.join(filtered_words)

            # Append the filtered sentence to the list
            filtered_sentences.append(filtered_sentence)

    # Write the filtered sentences to the output file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(filtered_sentences))

# Usage example:
input_file = '../data/test_text.txt'  # Replace with your input file path
output_file = '../data/test_cleaned_text.txt'  # Replace with the desired output file path
stopwords_file = 'stopwords.txt'  # Replace with your stopwords file path

remove_stopwords(input_file, output_file, stopwords_file)


input_file_path = output_file

# Specify the path where you want to save the lowercased text
output_file_path = output_file

# Open the input file for reading
with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Read the content of the file
    text = input_file.read()

# Convert the text to lowercase
lowercased_text = text.lower()

# Open the output file for writing
with open(output_file_path, "w", encoding="utf-8") as output_file:
    # Write the lowercased text to the output file
    output_file.write(lowercased_text)





import string

# Specify the path to your input text file
input_file_path = output_file_path

# Specify the path where you want to save the text without punctuation
output_file_path = output_file_path

# Function to remove punctuation from text
def remove_punctuation(text):
    # Create a translation table to remove punctuation characters
    translator = str.maketrans('', '', string.punctuation)
    
    # Use translate to remove punctuation
    text_without_punctuation = text.translate(translator)
    
    return text_without_punctuation

# Open the input file for reading
with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Read the content of the file
    text = input_file.read()

# Remove punctuation from the text
text_without_punctuation = remove_punctuation(text)

# Open the output file for writing
with open(output_file_path, "w", encoding="utf-8") as output_file:
    # Write the text without punctuation to the output file
    output_file.write(text_without_punctuation)