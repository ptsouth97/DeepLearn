#!/usr/bin/python3

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np


# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminderstats.csv')

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

y = np.array(df_region['life'])
X = np.array(df_region.drop('life', axis=1))

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print('CROSS VALIDATION SCORES:')
print(ridge_cv)
