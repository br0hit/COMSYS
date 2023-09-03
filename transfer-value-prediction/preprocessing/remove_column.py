import pandas as pd

column_dropped = "Name"
save_path = 

df = pd.read_csv('../data/train.csv')

# List of columns to remove
columns_to_remove = ["Value at beginning of 2020/21 season", "Value at beginning of 2021/22 season"]

# Drop the specified columns
df = df.drop(columns_to_remove, axis=1)

# Save the modified dataset to a new file
df.to_csv("../data/train_2021na.csv", index=False)

