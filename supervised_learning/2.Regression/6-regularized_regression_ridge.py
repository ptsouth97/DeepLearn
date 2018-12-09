#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


def main():
	''' Linear regression minimizes a loss function
		It choose a coefficient for each feature variable
		Large coefficients can lead to overfitting
		Regularization is penalizing large coefficients
		This function uses RIDGE REGRESSION'''

	# Create a dataframe from the .csv file
	df = pd.read_csv('gapminderstats.csv')

	# Create an array for the target variable
	y = np.array(df['life'])

	# Drop the target variable column from the data frame
	df_X = df.drop('life', axis=1)

	# Get the column names
	# df_columns = df_X.dtypes.index

	# Create an array for the features
	X = np.array(df_X)

	# Setup the array of alphas and lists to store scores
	alpha_space = np.logspace(-4, 0, 50)
	ridge_scores = []
	ridge_scores_std = []

	# Create a ridge regressor: ridge
	ridge = Ridge(normalize=True)

	# Compute scores over range of alphas
	for alpha in alpha_space:

		# Specify the alpha value to use: ridge.alpha
		ridge.alpha = alpha
    
		# Perform 10-fold CV: ridge_cv_scores
		ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
		# Append the mean of ridge_cv_scores to ridge_scores
		ridge_scores.append(np.mean(ridge_cv_scores))
    
		# Append the std of ridge_cv_scores to ridge_scores_std
		ridge_scores_std.append(np.std(ridge_cv_scores))

	# Display the plot
	display_plot(ridge_scores, ridge_scores_std, alpha_space)	


def display_plot(cv_scores, cv_scores_std, alpha_space):
	'''creates the plot'''

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(alpha_space, cv_scores)

	std_error = cv_scores_std / np.sqrt(10)

	ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
	ax.set_ylabel('CV Score +/- Std Error')
	ax.set_xlabel('Alpha')
	ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
	ax.set_xlim([alpha_space[0], alpha_space[-1]])
	ax.set_xscale('log')
	plt.show()


if __name__ == '__main__':
	main()
