#!/usr/bin/python3

# Import necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


def main():
	'''main function for testing'''

	# Read the file
	file_name = 'diabetes.csv'
	df = pd.read_csv(file_name)

	# Slice the feature (y) and target (X) arrays
	y = np.array(df['Outcome'])
	X = np.array(df.drop('Outcome', axis=1))

	random_search(X, y)


def random_search(X, y):
	''' Hyperparameter: variable (like k in k-NN) that must be chosen ahead of time
        How to choose best hyperparamter? Choose a bunch and evaluate performance
		RandomizedSearchCV tries a fixed number of hyperparameter values and is 
        less computationally expensive than GridSearchCV'''

	# Setup the parameters and distributions to sample from: param_dist
	param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}

	# Instantiate a Decision Tree classifier: tree
	tree = DecisionTreeClassifier()

	# Instantiate the RandomizedSearchCV object: tree_cv
	tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

	# Fit it to the data
	tree_cv.fit(X, y)

	# Print the tuned parameters and score
	print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
	print("Best score is {}".format(tree_cv.best_score_))

	return


if __name__ == '__main__':
	main()
