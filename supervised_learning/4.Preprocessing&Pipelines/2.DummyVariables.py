#!/usr/bin/python3

import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminderstats.csv')

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)