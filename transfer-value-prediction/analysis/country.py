import pandas as pd

# Your dataset
path = '../data/train.csv'

# Create a DataFrame
df = pd.read_csv(path)

# Count the number of unique countries
unique_countries = df["Country"].nunique()
print("Number of unique countries:", unique_countries)

# Use value_counts() to count the occurrences of each country in the 'Country' column
country_counts = df['Country'].value_counts()
# Display the number of rows corresponding to each country
print(country_counts)
# Create a DataFrame from the country counts
country_counts = pd.DataFrame({'Country': country_counts.index, 'Count': country_counts.values})
# Specify the file path where you want to save the counts
output_file_path = 'country_counts.csv'
# Save the counts to a CSV file
country_counts.to_csv(output_file_path, index=False)