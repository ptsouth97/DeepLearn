#!/usr/bin/python3

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import clusters, clustering2Dpoints


def main():
	''' create a KMeans model to find 3 clusters, and fit it to the data '''

	# Create array of data
	points, new_points = clusters.make_points()

	# Assign the columns of new_points: xs and ys
	xs = new_points[:,0]
	ys = new_points[:,1]

	# Get the labels and the model
	labels, model = clustering2Dpoints.get_labels()

	# Make a scatter plot of xs and ys, using labels to define the colors
	plt.scatter(xs, ys, c=labels, alpha=0.5)

	# Assign the cluster centers: centroids
	centroids = model.cluster_centers_

	# Assign the columns of centroids: centroids_x, centroids_y
	centroids_x = centroids[:,0]
	centroids_y = centroids[:,1]

	# Make a scatter plot of centroids_x and centroids_y
	plt.scatter(centroids_x, centroids_y, marker='D', s=50)
	plt.show()


if __name__ == '__main__':
	main()
