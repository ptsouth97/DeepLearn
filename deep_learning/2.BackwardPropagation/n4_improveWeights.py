#!/usr/bin/python3

import numpy as np


def main():
	''' Main function'''

	weights = np.array([0, 2, 1])
	input_data = np.array([1, 2, 3])
	target = 0

	update_weights(input_data, weights, target)
	


def update_weights(input_data, weights, target):
	''' Calculates slope of the loss function with respect to the target'''

	# Set the learning rate: learning_rate
	learning_rate = 0.01

	# Calculate the predictions: preds
	preds = (weights * input_data).sum()

	# Calculate the error: error
	error = preds - target

	# Calculate the slope: slope
	slope = input_data * error * 2

	# Update the weights: weights_updated
	weights_updated = weights - learning_rate * slope

	# Get updated predictions: preds_updated
	preds_updated = (weights_updated * input_data).sum()

	# Calculate updated error: error_updated
	error_updated = preds_updated - target

	# Print the original error
	print(error)

	# Print the updated error
	print(error_updated)

	
if __name__ == '__main__':
	main()
