#!/usr/bin/python3

import numpy as np


def main():
	''' Main function for testing purposes'''

	# Define inputs
	input_data = np.array([3, 5])

	# Define weights
	weights = {'node_0': np.array([2, 4]), 
               'node_1': np.array([ 4, -5]), 
               'output': np.array([2, 7])}

	# Calculate node 0 value: node_0_value
	node_0_value = (input_data * weights['node_0']).sum()

	# Calculate node 1 value: node_1_value
	node_1_value = (input_data * weights['node_1']).sum()

	# Put node values into array: hidden_layer_outputs
	hidden_layer_outputs = np.array([node_0_value, node_1_value])

	# Calculate output: output
	output = (hidden_layer_outputs * weights['output']).sum()

	# Print output
	print(output)


if __name__ == '__main__':
	main()
