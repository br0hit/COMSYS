import csv

with open('../data/train.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header row (assuming it contains column names)
    next(csv_reader)
    
    # Create separate text files for each column
    text_file = open('../data/train_text.txt', 'w')
    # label_file = open('labels.txt', 'w')

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Write the values from the desired columns to their respective text files
        text_file.write(row[1] + '\n')  # Age is the second column (index 1)
        # label_file.write(row[2] + '\n')  # Location is the third column (index 2)

    # Close the text files

    text_file.close()
    # label_file.close()



# Open the CSV file for reading
with open('../data/test.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header row (assuming it contains column names)
    next(csv_reader)
    
    # Create separate text files for each column
    text_file = open('../data/test_text.txt', 'w')
    # label_file = open('labels.txt', 'w')

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Write the values from the desired columns to their respective text files
        text_file.write(row[1] + '\n')  # Age is the second column (index 1)
        # label_file.write(row[2] + '\n')  # Location is the third column (index 2)

    # Close the text files

    text_file.close()
    # label_file.close()

print("train and test text seperated")

with open('../data/train.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header row (assuming it contains column names)
    next(csv_reader)
    
    # Create separate text files for each column
    text_file = open('../data/encoded_labels.txt', 'w')
    # label_file = open('labels.txt', 'w')

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Write the values from the desired columns to their respective text files
        text_file.write(row[2] + '\n')  # Age is the second column (index 1)
        # label_file.write(row[2] + '\n')  # Location is the third column (index 2)

    # Close the text files

    text_file.close()
    # label_file.close()


print('labels seperated!')







# Define the label-to-integer mapping
label_mapping = {
    'Anger/ Intermittent Explosive Disorder': 0,
    'Anxiety Disorder': 1,
    'Depression': 2,
    'Narcissistic Disorder': 3,
    'Panic Disorder': 4
}

# Open the input labels file for reading
with open('../data/labels.txt', 'r') as labels_file:
    labels = labels_file.read().splitlines()

# Create a list to store the encoded labels
encoded_labels = []

# Encode the labels based on the mapping
for label in labels:
    encoded_label = label_mapping.get(label)
    if encoded_label is not None:
        encoded_labels.append(encoded_label)

# # Print the encoded labels
# print(encoded_labels)

# Optionally, save the encoded labels to a new file
with open('../data/encoded_labels.txt', 'w') as output_file:
    for encoded_label in encoded_labels:
        output_file.write(str(encoded_label) + '\n')

print('Label encoding completed!')



