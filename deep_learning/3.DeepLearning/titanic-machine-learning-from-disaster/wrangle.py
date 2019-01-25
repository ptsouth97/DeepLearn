#!/usr/bin/python3

import pandas as pd

df = pd.read_csv('changes.csv')
df = df.drop(['PassengerId', 'embarked', 'sex'], axis=1)
df.to_csv('titanic.csv')
