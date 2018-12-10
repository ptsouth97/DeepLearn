#!/usr/bin/python3

# Import necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
	'''main function for testing'''

	# Read the file
	file_name = 'diabetes.csv'
	df = pd.read_csv(file_name)

	# Slice the feature (y) and target (X) arrays
	y = np.array(df['Outcome'])
	X = np.array(df.drop('Outcome', axis=1))

	# Create training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	auc(X, y, X_train, X_test, y_train, y_test)


def auc(X, y, X_train, X_test, y_train, y_test):
	''' Area Under Curve (AUC) computation'''

	# Create the classifier: logreg
	logreg = LogisticRegression()	

	# Fit the classifier to the training data
	logreg.fit(X_train, y_train)

	# Compute predicted probabilities: y_pred_prob
	y_pred_prob = logreg.predict_proba(X_test)[:,1]

	# Compute and print AUC score
	print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

	# Compute cross-validated AUC scores: cv_auc
	cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

	# Print list of AUC scores
	print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

	return


if __name__ == '__main__':
	main()
