import pandas as pd

# Read the original CSV file
data_path = '../data/test.csv'
df = pd.read_csv(data_path)

# Find rows with missing values
missing_rows = df[df.isnull().any(axis=1)]

# Create a new DataFrame to store information about missing columns for each ID
missing_info_df = pd.DataFrame(columns=['id', 'missing_columns'])

# Iterate through each ID and store the missing column names
for idx, row in missing_rows.iterrows():
    missing_columns = row.index[row.isnull()].tolist()
    missing_info_df = missing_info_df.append({'id': row['id'], 'missing_columns': missing_columns}, ignore_index=True)

# Save the DataFrame with missing data info to a new CSV file
missing_info_df.to_csv('../results/missing_data_test.csv', index=False)
