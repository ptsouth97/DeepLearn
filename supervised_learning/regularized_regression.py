#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso


def main():
	''' Linear regression minimizes a loss function
		It choose a coefficient for each feature variable
		Large coefficients can lead to overfitting
		Regularization is penalizing large coefficients
		This function uses RIDGE REGRESSION'''

	# Create a dataframe from the .csv file
	df = pd.read_csv('gapminderstats.csv')

	# Create an array for the target variable
	y = np.array(df['life'])

	# Drop the target variable column from the data frame
	df_X = df.drop('life', axis=1)

	# Get the column names
	df_columns = df_X.dtypes.index

	# Create an array for the features
	X = np.array(df_X)

	# Instantiate a lasso regressor: lasso
	lasso = Lasso(alpha=0.4, normalize=True)

	# Fit the regressor to the data
	lasso.fit(X, y)

	# Compute and print the coefficients
	lasso_coef = lasso.coef_
	print(lasso_coef)

	# Plot the coefficients
	plt.plot(range(len(df_columns)), lasso_coef)
	plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
	plt.margins(0.02)
	plt.show()
	

if __name__ == '__main__':
	main()
