#!/usr/bin/python3

# Import necessary modules
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


def main():
	''' the main function'''

	file_name = 'gapminderstats.csv'
	df = pd.read_csv(file_name)

	y = np.array(df['life'])
	X = np.array(df.drop('life', axis=1))

	print(np.shape(X))
	print(np.shape(y))

	# Create training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

	# Create the regressor: reg_all
	reg_all = LinearRegression()

	# Fit the regressor to the training data
	reg_all.fit(X_train, y_train)

	# Predict on the test data: y_pred
	y_pred = reg_all.predict(X_test)

	# Compute and print R^2 and RMSE
	print("R^2: {}".format(reg_all.score(X_test, y_test)))
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	print("Root Mean Squared Error: {}".format(rmse))


if __name__ == '__main__':
	main()
