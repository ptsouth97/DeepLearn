#!/usr/bin/python3

from sklearn.neighbors import KNeighborsClassifier
import load_data
import pandas as pd


def main():
	''' main function'''

	the_file = 'votes.csv'
	df = load_data.votes(the_file)

	# Create arrays for the features and the response variable
	y = df['party'].values
	X = df.drop('party', axis=1).values

	# Create a k-NN classifier with 6 neighbors
	knn = KNeighborsClassifier(n_neighbors=6)

	# Fit the classifier to the data
	knn.fit(X, y)

	# Predict the labels for the training data X
	y_pred = knn.predict(X)

	# Make a new data point X_new to test
	X_new = pd.DataFrame([0.041569, 0.717944, 0.108728, 0.417277, \
  						  0.414185, 0.397297, 0.117327, 0.306765, \
						  0.988804, 0.358076, 0.667909, 0.019063, \
						  0.079745, 0.809151, 0.343293, 0.889644]).T   

	# Predict and print the label for the new data point X_new
	new_prediction = knn.predict(X_new)
	print("Prediction: {}".format(new_prediction))


if __name__ == '__main__':
	main()
