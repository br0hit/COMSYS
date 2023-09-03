import pandas as pd

df = pd.read_csv('../data/train.csv')
df = df.drop("Value at beginning of 2020/21 season", axis=1)

# Save the modified datasets to new files
df.to_csv("../data/train_2020na.csv", index=False)