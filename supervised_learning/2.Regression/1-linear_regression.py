#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


def main():
	'''main function for testing purposes'''

	# Read the data file and convert to dataframe
	filename = 'life expectancy.csv'
	df = pd.read_csv(filename)

	X = df['fertility'].values.reshape(-1, 1)
	y = df['life expectancy'].values

	linreg(y, X)


def linreg(y, X_fertility):
	'''performs and plots linear regression for countries life expectancy vs fertility for GapMinder Data year 2008'''

	# Create the regressor: reg
	reg = LinearRegression()

	# Create the prediction space
	prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

	# Fit the model to the data
	reg.fit(X_fertility, y)

	# Compute predictions over the prediction space: y_pred
	y_pred = reg.predict(prediction_space)

	# Print R^2 
	print(reg.score(X_fertility, y))

	# Plot regression line
	plt.scatter(X_fertility, y)
	plt.plot(prediction_space, y_pred, color='black', linewidth=3)
	plt.xlabel('fertility')
	plt.ylabel('Life expectancy')
	plt.show()


if __name__ == '__main__':
	main()
