#!/usr/bin/python3

# Import necessary modules
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np


def main():
	''' cross validation is a protection against arbitrary splits in the data set that may skew R^2.  It splits the data
		into k-folds and holds out one fold at a time as a test.  The model is fit on the remaining folds and predicts on
		the test set and compute the metric of interest. More folds is more computationally expensive'''

	file_name = 'gapminderstats.csv'
	df = pd.read_csv(file_name)

	y = np.array(df['life'])
	X = np.array(df.drop('life', axis=1))

	print(np.shape(X))
	print(np.shape(y))

	# Create the regressor: reg_all
	reg = LinearRegression()

	# Compute 5-fold cross-validation scores: cv_scores
	cv_scores = cross_val_score(reg, X, y, cv=5)

	# Print the 5-fold cross-validation scores
	print(cv_scores)

	print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


if __name__ == '__main__':
	main()
