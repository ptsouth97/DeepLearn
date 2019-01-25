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


def relu(input):
	'''Define your relu activation function here'''

	# Calculate the value for the output of the relu function: output
	output = max(input, 0)

	# Return the value just calculated
	return(output)


def predict_with_network(input_data):
	''' predicts...'''

    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

	# Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)


if __name__ == '__main__':
	main()
