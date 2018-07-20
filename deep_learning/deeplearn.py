#!/usr/bin/python3

from keras.layers import Dense
from keras.models import Sequential
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import time


def main():
	''' main function for testing purposes'''

	# Use MNIST dataset:  http://yann.lecun.com/exdb/mnist/
	# 28 x 28 grid flattened to 784 values for each image (flattened to 784 by 1 array)
	# only use 2,500 images rather than 60,000

	img, lab = load_mnist()

	show_image(img)

	# model(img, lab)


def load_mnist():
	''' loads mnist data'''

	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

	images = mnist.train.images
	images = images[0:2500]
	
	labels = mnist.train.labels
	labels = labels[0:2500]

	print(labels[2])
	
	return images, labels


def show_image(images):
	''' displays MNIST image'''
	
	for i in range(0, 3):

		this_image = images[i]
		this_image = np.array(this_image, dtype='float')
		pixels = this_image.reshape((28, 28))

		fig = plt.figure()
		plt.imshow(pixels, cmap='gray')
		plt.show(block=False)

		time.sleep(2)
		plt.close(fig)

	return


def model(X, y):
	'''create and fit a basic model'''

	# Create the model: model
	model = Sequential()

	# Add the first hidden layer
	model.add(Dense(50, activation='relu', input_shape=(784,)))

	# Add the second hidden layer
	model.add(Dense(50, activation='relu', input_shape=(784,)))

	# Add the output layer
	model.add(Dense(10, activation='softmax'))

	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Fit the model
	model.fit(X, y, validation_split=0.3)

	return


if __name__ == '__main__':
	main()
