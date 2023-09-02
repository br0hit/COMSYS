import pandas as pd
import matplotlib.pyplot as plt

path = '../data/train.csv'

# Read your dataset from a CSV file or any other format
# Replace 'your_dataset.csv' with the actual file path
df = pd.read_csv(path)

# Select the relevant columns for analysis
value_columns = ["Value at beginning of 2020/21 season", "Value at beginning of 2021/22 season", "Value at beginning of 2022/23 season", "Value at beginning of 2023/24 season"]

# Calculate the changes in value for each player over the seasons
for i in range(1, len(value_columns)):
    df[f"Change {value_columns[i-1]} to {value_columns[i]}"] = df[value_columns[i]] - df[value_columns[i-1]]

# Create a scatter plot to visualize the changes
plt.figure(figsize=(12, 8))
plt.scatter(df["Change Value at beginning of 2020/21 season to Value at beginning of 2021/22 season"],
            df["Change Value at beginning of 2021/22 season to Value at beginning of 2022/23 season"],
            c=df["Change Value at beginning of 2022/23 season to Value at beginning of 2023/24 season"],
            cmap='coolwarm', alpha=0.8)
plt.title("Player Value Changes Over Multiple Seasons")
plt.xlabel("Change 2020/21 to 2021/22")
plt.ylabel("Change 2021/22 to 2022/23")
plt.colorbar(label="Change 2022/23 to 2023/24")
plt.grid(True)

# Add player names as labels (optional)
for i, row in df.iterrows():
    plt.annotate(row["id"], (row["Change Value at beginning of 2020/21 season to Value at beginning of 2021/22 season"],
                             row["Change Value at beginning of 2021/22 season to Value at beginning of 2022/23 season"]))

plt.show()
