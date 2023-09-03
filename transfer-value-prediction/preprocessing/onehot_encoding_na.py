import pandas as pd

data_path = "../data/train_2021na.csv"

data = pd.read_csv(data_path)
data.dropna(inplace=True)

# Replace 'data' with your actual dataset and 'country_column' with the column containing country names
threshold = 98
common_countries = data['Country'].value_counts()[data['Country'].value_counts() >= threshold].index.tolist()

# Create a new column with country names, but replace less common countries with 'Other'
data['Country_encoded'] = data['Country'].apply(lambda x: x if x in common_countries else 'Other')

# Perform one-hot encoding
encoded_data = pd.get_dummies(data, columns=['Country_encoded'], prefix='Country')

# # Drop the original 'country_column' and 'encoded_country' columns if needed
# encoded_data = encoded_data.drop(['country_column', 'encoded_country'], axis=1)

# print(encoded_data)
encoded_data.to_csv(f"{data_path}_onehotEncoded{threshold}.csv")
