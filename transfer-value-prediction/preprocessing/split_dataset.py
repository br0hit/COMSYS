import pandas as pd

# Read the original CSV file
data_path = '../data/test.csv'
df = pd.read_csv(data_path)

# Split the dataset into two parts based on missing values
complete_data_df = df.dropna()
missing_data_df = df[df.isnull().any(axis=1)]

# Save the two parts to separate CSV files
complete_data_df.to_csv('../data/test_full.csv', index=False)
missing_data_df.to_csv('../data/test_missing.csv', index=False)
