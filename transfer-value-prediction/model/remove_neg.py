import pandas as pd

# Step 1: Read the CSV files into DataFrames
file1 = 'predictions_stacking.csv'  # Replace with the actual file paths
file2 = 'predictions_stacking_nonneg.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Step 2: Create a new DataFrame with the 'id' column
result_df = pd.DataFrame(df1['id'])

# Step 3: Use conditions to assign 'label' values
result_df['label'] = df1.apply(lambda row: row['label'] if row['label'] > 0 else df2.loc[df2['id'] == row['id'], 'label'].values[0], axis=1)

# Step 4: Save the new DataFrame to a new CSV file
result_df.to_csv('result.csv', index=False)