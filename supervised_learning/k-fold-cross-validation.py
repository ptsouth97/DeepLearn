#!/usr/bin/python3

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


def main():
	''' main function'''

	df = pd.read_csv('gapminderstats.csv')

	y = np.array(df['life'])
	X = np.array(df.drop('life', axis=1))

	# Create a linear regression object: reg
	reg = LinearRegression()

	# Perform 3-fold CV
	cvscores_3 = cross_val_score(reg, X, y, cv=3)
	print(np.mean(cvscores_3))

	# Perform 10-fold CV
	cvscores_10 = cross_val_score(reg, X, y, cv=10)
	print(np.mean(cvscores_10))


if __name__ == '__main__':
	main()



