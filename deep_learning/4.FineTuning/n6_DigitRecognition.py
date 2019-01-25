#!/usr/bin/python3

#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


def main():
	''' Main function for testing purposes'''

	# Load data from MNIST dataset
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	# Slice the first 2500 of 60000 images
	X_train = X_train[0:2500]
	y_train = y_train[0:2500]

	# get the first image and it's label
	img1_arr, img1_label = X_train[0], y_train[0]
	#print(img1_arr.shape, img1_label)

	#show_image(img1_arr)

	# Flatten the images from 28x28
	X_train = X_train.reshape(X_train.shape[0], 784)

	# Convert data type
	X_train = X_train.astype('float32')

	# Normalize values
	X_train /= 255

	# One hot encode the labels
	y_train = to_categorical(y_train)

	digits(X_train, y_train)


def show_image(img1_arr):
	''' Shows digit image'''

	# reshape first image(1 D vector) to 2D dimension image
	img1_2d = np.reshape(img1_arr, (28, 28))

	# show it
	plt.subplot(111)
	plt.imshow(img1_2d, cmap=plt.get_cmap('gray'))
	plt.show()

	return


def digits(X, y):
	''' Build model to recognize handwritten digits '''

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
	model.fit(X, y, validation_split=0.3, epochs=10)


if __name__ == '__main__':
	main()
