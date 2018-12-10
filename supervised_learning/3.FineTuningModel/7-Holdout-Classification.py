#!/usr/bin/python3

# Import necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression


def main():
	'''main function for testing'''

	# Read the file
	file_name = 'diabetes.csv'
	df = pd.read_csv(file_name)

	# Slice the feature (y) and target (X) arrays
	y = np.array(df['Outcome'])
	X = np.array(df.drop('Outcome', axis=1))

	hold_out_classification(X, y)


def hold_out_classification(X, y):
	''' Hold-out set reasoning: if all data is used for cross-validation, estimating model
        performance on any of it may not provide an accurate picture of how it will perform
        on unseen data. Therefore, split data into training and hold-out set. Perform grid
        search cross validation on training set. Choose best hyperparameters and evaluate on
        hold-out set.  '''

	# Create the hyperparameter grid
	c_space = np.logspace(-5, 8, 15)
	param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

	# Instantiate the logistic regression classifier: logreg
	logreg = LogisticRegression()

	# Create train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	# Instantiate the GridSearchCV object: logreg_cv
	logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

	# Fit it to the training data
	logreg_cv.fit(X_train, y_train)

	# Print the optimal parameters and best score
	print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
	print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

	return


if __name__ == '__main__':
	main()
