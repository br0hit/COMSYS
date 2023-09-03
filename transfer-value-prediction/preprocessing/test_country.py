import pandas as pd

df = pd.read_csv("../data/test.csv")

df =df.drop("Country", axis=1)

# Create a list of the desired column IDs
desired_ids = [408, 1643, 892, 453, 764, 1696, 548, 67, 1681, 772, 170, 184, 1518, 593, 522, 309, 1456]

# Create a new test set containing only the rows with the desired "id" values
new_test_set = df[df['id'].isin(desired_ids)]



new_test_set.to_csv("../data/test_nocountry.csv", index=False)
