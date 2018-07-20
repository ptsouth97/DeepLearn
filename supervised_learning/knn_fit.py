#!/usr/bin/python3.5

from sklearn.neighbors import KNeighborsClassifier
import load_data


def main():
	''' main function'''

	df = load_data.main()

	# Create arrays for the features and the response variable
	y = df['party'].values
	X = df.drop('party', axis=1).values

	# Create a k-NN classifier with 6 neighbors
	knn = KNeighborsClassifier(n_neighbors=6)

	# Fit the classifier to the data
	knn.fit(X, y)


if __name__ == '__main__':
	main()
