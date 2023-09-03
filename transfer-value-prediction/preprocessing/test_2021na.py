import pandas as pd

df = pd.read_csv("../data/test.csv")

# Create a list of the desired column IDs
desired_ids = [765, 1649, 49, 618, 694, 844, 679, 506, 420,1647]

# Create a new test set containing only the rows with the desired "id" values
new_test_set = df[df['id'].isin(desired_ids)]


new_test_set.to_csv("../data/test_2021na.csv", index=False)
