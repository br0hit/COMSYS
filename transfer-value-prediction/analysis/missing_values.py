import pandas as pd

# Your dataset
path = '../data/train.csv'

# Create a DataFrame
df = pd.read_csv(path)

# Count the number of missing values in each column
missing_values = df.isnull().sum()

# Display the missing values count for each column
print(missing_values)