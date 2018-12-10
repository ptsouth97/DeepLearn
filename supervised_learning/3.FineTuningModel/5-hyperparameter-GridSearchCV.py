#!/usr/bin/python3

# Import necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def main():
	'''main function for testing'''

	# Read the file
	file_name = 'diabetes.csv'
	df = pd.read_csv(file_name)

	# Slice the feature (y) and target (X) arrays
	y = np.array(df['Outcome'])
	X = np.array(df.drop('Outcome', axis=1))

	grid_search(X, y)


def grid_search(X, y):
	''' Hyperparameter: variable (like k in k-NN) that must be chosen ahead of time
        How to choose best hyperparamter? Choose a bunch and evaluate performance
        GridSearchCV checks over a defined set of parameters. Best option, but most
        computationally expensive. Logistic regression also has a regularization 
        parameter: C. C controls the inverse of the regularization strength, and this 
        is what you will tune in this exercise. A large C can lead to an overfit model, 
        while a small C can lead to an underfit model.'''

	# Setup the hyperparameter grid
	c_space = np.logspace(-5, 8, 15)
	param_grid = {'C': c_space}

	# Create the classifier: logreg
	logreg = LogisticRegression()	

	# Instantiate the GridSearchCV object: logreg_cv
	logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
	
	# Fit it to the data
	logreg_cv.fit(X, y)

	# Print the tuned parameters and score
	print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
	print("Best score is {}".format(logreg_cv.best_score_))

	return


if __name__ == '__main__':
	main()
