import pandas as pd

df = pd.read_csv("../data/test.csv")

# Create a list of the desired column IDs
desired_ids = [1292, 438, 567, 1225, 1085, 426, 613, 237, 650, 240, 1196, 1494, 1648, 493, 620, 411]

# Create a new test set containing only the rows with the desired "id" values
new_test_set = df[df['id'].isin(desired_ids)]


new_test_set.to_csv("../data/test_2020na.csv", index=False)
