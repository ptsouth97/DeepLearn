#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def main():
	''' Main function'''

	weights = np.array([0, 2, 1])
	input_data = np.array([1, 2, 3])
	target = 0

	n_updates = 20
	mse_hist = []

	# Iterate over the number of updates
	for i in range(n_updates):

		# Calculate the slope: slope
		slope = get_slope(input_data, target, weights)
    
		# Update the weights: weights
		weights = weights - 0.01 * slope
    
		# Calculate mse with new weights: mse
		mse = get_mse(input_data, target, weights)
    
		# Append the mse to mse_hist
		mse_hist.append(mse)
	
	# Plot the mse history
	plt.plot(mse_hist)
	plt.xlabel('Iterations')
	plt.ylabel('Mean Squared Error')
	plt.show()	


def get_slope(input_data, target, weights):
	''' Calculates slope of the loss function with respect to the target'''

	# Calculate the predictions: preds
	preds = (weights * input_data).sum()

	# Calculate the error: error
	error = preds - target

	# Calculate the slope: slope
	slope = input_data * error * 2

	return slope


def get_mse(input_data, target, weights):
	''' Calculates the mean squared error'''

	# Get updated predictions: preds_updated
	preds_updated = (weights * input_data).sum()

	# Calculate updated error: error_updated
	error_updated = preds_updated - target

	mse = error_updated**2

	return mse

	
if __name__ == '__main__':
	main()
