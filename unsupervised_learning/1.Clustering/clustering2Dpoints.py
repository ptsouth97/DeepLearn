#!/usr/bin/python3

from sklearn.cluster import KMeans
import clusters


def main():
	''' Main function for testing and printing labels '''

	labels = get_labels()
	print(labels)


def get_labels():
	''' Create a KMeans model to find 3 clusters, and fit it to the data.  Returns labels and model'''

	# Create array of data
	points, new_points = clusters.make_points()

	# Create a KMeans instance with 3 clusters: model
	model = KMeans(n_clusters=3)

	# Fit model to points
	model.fit(points)

	# Determine the cluster labels of new_points: labels
	labels = model.predict(new_points)

	return labels, model


if __name__ == '__main__':
	main()
