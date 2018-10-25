#!/usr/bin/python3

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
 
 
def main():
	'''Evaluates the performance of a binary classifier by computing a confusion matrix and generating a classification report'''
 
	file_name = 'diabetes.csv'
	df = pd.read_csv(file_name)
 
	y = np.array(df['Outcome'])
	X = np.array(df.drop('Outcome', axis=1))

	# Create training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	# Create the classifier: logreg
	logreg = LogisticRegression()

	# Fit the classifier to the training data
	logreg.fit(X_train, y_train)

	# Predict the labels of the test set: y_pred
	y_pred = logreg.predict(X_test)

	# Compute and print the confusion matrix and classification report
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred))


if __name__ == '__main__':
	main()
