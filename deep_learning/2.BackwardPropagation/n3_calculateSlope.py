#!/usr/bin/python3

import numpy as np


def main():
	''' Main function'''

	weights = np.array([0, 2, 1])
	input_data = np.array([1, 2, 3])
	target = 0

	calc_slope(input_data, weights, target)
	


def calc_slope(input_data, weights, target):
	''' Calculates slope of the loss function with respect to the target'''

	# Calculate the predictions: preds
	preds = (weights * input_data).sum()

	# Calculate the error: error
	error = preds - target

	# Calculate the slope: slope
	slope = input_data * error * 2

	# Print the slope
	print(slope)

	
if __name__ == '__main__':
	main()
