import pandas as pd

df = pd.read_csv('../data/train.csv')
df = df.drop("Country", axis=1)

# Save the modified datasets to new files
df.to_csv("../data/train_nocountry.csv", index=False)