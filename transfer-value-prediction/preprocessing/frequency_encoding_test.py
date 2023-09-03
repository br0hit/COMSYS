import pandas as pd

path = '../data/test_full.csv'

# Load the CSV file with the encoding
encoding_df = pd.read_csv('../data/encoding.csv')  # Replace 'encoding.csv' with your file path

df = pd.read_csv(path)

# Merge the two DataFrames based on the "Country" column
result_df = df.merge(encoding_df, on='Country', how='left')

# Rename the "Count" column to "Country_encoded"
result_df.rename(columns={'Count': 'Country_encoded'}, inplace=True)

# Rearrange the columns as per your specified order
new_column_order = [
    'id',
    'Aerial Duels won',
    'Age',
    'Assists',
    'Attacking options created',
    'Attempted Passes',
    'Blocks',
    'Clearances',
    'Country',
    'Country_encoded',
    'Expected Goal Contributions',
    'Interceptions',
    'Open Play Goals',
    'Open Play Expected Goals',
    'Percentage of Passes Completed',
    'Progressive Passes Rec',
    'Progressive Passes',
    'Progressive Carries',
    'Shots',
    'Successful Dribbles',
    'Touches in attacking penalty area',
    'Tackles',
    'Value at beginning of 2020/21 season',
    'Value at beginning of 2021/22 season',
    'Value at beginning of 2022/23 season',
]

result_df = result_df[new_column_order]

result_df.to_csv("../data/test_full_freqEncoded.csv")
