#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale, Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('winequality-white.csv', sep=';')

# Create feature array
X = df.drop('quality', axis=1).values

# Create target array
y = df['quality'].values

# Convert quality column based on >< 5
y[y <= 5] = 1
y[y > 5] = 0
y = y.astype(dtype=bool)

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))
