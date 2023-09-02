import pandas as pd

df = pd.read_csv('../data/train_initial.csv')
df = df.drop("Name", axis=1)

# Save the modified datasets to new files
df.to_csv("../data/train.csv", index=False)