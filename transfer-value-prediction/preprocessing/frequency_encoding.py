import pandas as pd

# Sample DataFrame with a 'country' column
data_path = '../data/train_onehotEncoded50.csv'
save_path = '../data/train_onehotEncoded50_freq.csv'

df = pd.read_csv(data_path)

# Calculate the frequency of each category in the 'country' column
frequency_map = df['Country'].value_counts().to_dict()

# Create a new column 'country_encoded' using the frequency mapping
df['Country_encoded'] = df['Country'].map(frequency_map)


# Rearrange the columns as per your specified order
new_column_order = [
    'id',
    'Aerial Duels won',
    'Age',
    'Assists',
    'Attacking options created',
    'Attempted Passes',
    'Blocks',
    'Clearances',
    
    'Country',
    'Country_encoded',
    'Country_Spain',
    'Country_France',
    'Country_Germany',
    'Country_England',
    'Country_Italy',
    'Country_Brazil',
    'Country_Other',

    'Expected Goal Contributions',
    'Interceptions',
    'Open Play Goals',
    'Open Play Expected Goals',
    'Percentage of Passes Completed',
    'Progressive Passes Rec',
    'Progressive Passes',
    'Progressive Carries',
    'Shots',
    'Successful Dribbles',
    'Touches in attacking penalty area',
    'Tackles',
    'Value at beginning of 2020/21 season',
    'Value at beginning of 2021/22 season',
    'Value at beginning of 2022/23 season',
    'Value at beginning of 2023/24 season'
    
]

df = df[new_column_order]


df.to_csv(save_path)
