#!/usr/bin/python3

import numpy as np


def main():
	''' Main function'''

	# The data point you will make a prediction for
	input_data = np.array([0, 3])

	# Sample weights
	weights_0 = {'node_0': [2, 1],
                 'node_1': [1, 2],
                 'output': [1, 1]
                }

	# The actual target value, used to calculate the error
	target_actual = 3

	# Make prediction using original weights
	model_output_0 = predict_with_network(input_data, weights_0)

	# Calculate error: error_0
	error_0 = model_output_0 - target_actual

	# Create weights that cause the network to make perfect prediction (3): weights_1
	weights_1 = {'node_0': [2, 1],
                 'node_1': [1, 2],
                 'output': [-1, 1]
                }

	# Make prediction using new weights: model_output_1
	model_output_1 = predict_with_network(input_data, weights_1)

	# Calculate error: error_1
	error_1 = model_output_1 - target_actual

	# Print error_0 and error_1
	print(error_0)
	print(error_1)


def relu(input):
	'''Define your relu activation function here'''

	# Calculate the value for the output of the relu function: output
	output = max(input, 0)
    
	# Return the value just calculated
	return(output)


def predict_with_network(input_data, weights):
	''' Uses forward propagation to make a prediction'''

	# Calculate node 0 in the first hidden layer
	node_0_0_input = (input_data * weights['node_0']).sum()
	node_0_0_output = relu(node_0_0_input)
	
	# Calculate node 1 in the first hidden layer
	node_0_1_input = (input_data * weights['node_1']).sum()
	node_0_1_output = relu(node_0_1_input)

	# Put node values into array: hidden_0_outputs
	hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
	# Calculate model output: model_output
	model_output = (hidden_0_outputs * weights['output']).sum()
    
	# Return model_output
	return(model_output)


if __name__ == '__main__':
	main()
