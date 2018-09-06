#!/usr/bin/python3

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
	'''Evaluates the performance of a binary classifier by computing a confusion matrix and generating a classification report'''

	file_name = 'gapminderstats.csv'
	df = pd.read_csv(file_name)

	y = np.array(df['y'])
	X = np.array(df.drop('y', axis=1))

	# Create training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	# Instantiate a k-NN classifier: knn
	knn = KNeighborsClassifier(n_neighbors=6)

	# Fit the classifier to the training data
	knn.fit(X_train, y_train)

	# Predict the labels of the test data: y_pred
	y_pred = knn.predict(X_test)

	# Generate the confusion matrix and classification report
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred))


if __name__ == '__main__':
	main()
