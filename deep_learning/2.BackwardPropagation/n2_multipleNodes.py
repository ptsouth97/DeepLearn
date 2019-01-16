#!/usr/bin/python3

import numpy as np
from sklearn.metrics import mean_squared_error

def main():
	''' Main function'''

	# The data point you will make a prediction for
	input_data = np.array([[0, 3], [1, 2], [-1, -2], [4, 0]])

	# Sample weights
	weights_0 = {'node_0': [2, 1],
                 'node_1': [1, 2],
                 'output': [1, 1]
                }

	weights_1 = {'node_0': [2, 1],
                 'node_1': [1. , 1.5],
                 'output': [1. , 1.5]}

	target_actuals = [1, 3, 5, 7]

	# Create model_output_0 
	model_output_0 = []
	# Create model_output_0
	model_output_1 = []

	# Loop over input_data
	for row in input_data:
		# Append prediction to model_output_0
		model_output_0.append(predict_with_network(row, weights_0))
    
		# Append prediction to model_output_1
		model_output_1.append(predict_with_network(row, weights_1))

	# Calculate the mean squared error for model_output_0: mse_0
	mse_0 = mean_squared_error(target_actuals, model_output_0)

	# Calculate the mean squared error for model_output_1: mse_1
	mse_1 = mean_squared_error(target_actuals, model_output_1)

	# Print mse_0 and mse_1
	print("Mean squared error with weights_0: %f" %mse_0)
	print("Mean squared error with weights_1: %f" %mse_1)


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
