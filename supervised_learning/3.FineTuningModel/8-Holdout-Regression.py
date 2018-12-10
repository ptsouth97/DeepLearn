#!/usr/bin/python3

# Import necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


def main():
	'''main function for testing'''

	# Read the data file and convert to dataframe
	filename = 'gapminderstats.csv'
	df = pd.read_csv(filename)
 
	y = df['life'].values.reshape(-1, 1)
	X = df.drop('life', axis=1).values

	hold_out_regression(X, y)


def hold_out_regression(X, y):
	''' Hold-out set reasoning: if all data is used for cross-validation, estimating model
        performance on any of it may not provide an accurate picture of how it will perform
        on unseen data. Therefore, split data into training and hold-out set. Perform grid
        search cross validation on training set. Choose best hyperparameters and evaluate on
        hold-out set.  '''

	# Create train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	# Create the hyperparameter grid
	l1_space = np.linspace(0, 1, 30)
	param_grid = {'l1_ratio': l1_space}

	# Instantiate the ElasticNet regressor: elastic_net
	elastic_net = ElasticNet()

	# Setup the GridSearchCV object: gm_cv
	gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

	# Fit it to the training data
	gm_cv.fit(X_train, y_train)

	# Predict on the test set and compute metrics
	y_pred = gm_cv.predict(X_test)
	r2 = gm_cv.score(X_test, y_test)
	mse = mean_squared_error(y_test, y_pred)
	print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
	print("Tuned ElasticNet R squared: {}".format(r2))
	print("Tuned ElasticNet MSE: {}".format(mse))

	return


if __name__ == '__main__':
	main()
