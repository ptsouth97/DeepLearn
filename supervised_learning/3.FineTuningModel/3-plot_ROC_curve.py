#!/usr/bin/python3

# Import necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


def main():
	'''main function for testing'''

	file_name = 'diabetes.csv'
	df = pd.read_csv(file_name)

	y = np.array(df['Outcome'])
	X = np.array(df.drop('Outcome', axis=1))

	# Create training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	plot_roc_curve(X_train, X_test, y_train, y_test)


def plot_roc_curve(X_train, X_test, y_train, y_test):
	''' plot ROC curve'''

	# Create the classifier: logreg
	logreg = LogisticRegression()

	# Fit the classifier to the training data
	logreg.fit(X_train, y_train)

	# Compute predicted probabilities: y_pred_prob
	y_pred_prob = logreg.predict_proba(X_test)[:,1]

	# Generate ROC curve values: fpr, tpr, thresholds
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

	# Plot ROC curve
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.show()

	return


if __name__ == '__main__':
	main()
