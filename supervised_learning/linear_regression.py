#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def linreg(y, X_fertility):
	'''performs and plots linear regression'''

	# Create the regressor: reg
	reg = LinearRegression()

	# Create the prediction space
	prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

	# Fit the model to the data
	reg.fit(X_fertility, y)

	# Compute predictions over the prediction space: y_pred
	y_pred = reg.predict(prediction_space)

	# Print R^2 
	print(reg.score(X_fertility, y))

	# Plot regression line
	plt.scatter(X_fertility, y)
	plt.plot(prediction_space, y_pred, color='black', linewidth=3)
	plt.xlabel('fertility')
	plt.ylabel('Life expectancy')
	plt.show()


def main():
	''' main function'''

	y = np.array([[ 75.3],
       [ 58.3],
       [ 75.5],
       [ 72.5],
       [ 81.5],
       [ 80.4],
       [ 70.6],
       [ 72.2],
       [ 68.4],
       [ 75.3],
       [ 70.1],
       [ 79.4],
       [ 70.7],
       [ 63.2],
       [ 67.6],
       [ 70.9],
       [ 61.2],
       [ 73.9],
       [ 73.2],
       [ 59.4],
       [ 57.4],
       [ 66.2],
       [ 56.6],
       [ 80.7],
       [ 54.8],
       [ 78.9],
       [ 75.1],
       [ 62.6],
       [ 58.6],
       [ 79.7],
       [ 55.9],
       [ 76.5],
       [ 77.8],
       [ 78.7],
       [ 61. ],
       [ 74. ],
       [ 70.1],
       [ 74.1],
       [ 56.7],
       [ 60.4],
       [ 74. ],
       [ 65.7],
       [ 79.4],
       [ 81. ],
       [ 57.5],
       [ 62.2],
       [ 72.1],
       [ 80. ],
       [ 62.7],
       [ 79.5],
       [ 70.8],
       [ 58.3],
       [ 51.3],
       [ 63. ],
       [ 61.7],
       [ 70.9],
       [ 73.8],
       [ 82. ],
       [ 64.4],
       [ 69.5],
       [ 76.9],
       [ 79.4],
       [ 80.9],
       [ 81.4],
       [ 75.5],
       [ 82.6],
       [ 66.1],
       [ 61.5],
       [ 72.3],
       [ 77.6],
       [ 45.2],
       [ 61. ],
       [ 72. ],
       [ 80.7],
       [ 63.4],
       [ 51.4],
       [ 74.5],
       [ 78.2],
       [ 55.8],
       [ 81.4],
       [ 63.6],
       [ 72.1],
       [ 75.7],
       [ 69.6],
       [ 63.2],
       [ 73.3],
       [ 55. ],
       [ 60.8],
       [ 68.6],
       [ 80.3],
       [ 80.2],
       [ 75.2],
       [ 59.7],
       [ 58. ],
       [ 80.7],
       [ 74.6],
       [ 64.1],
       [ 77.1],
       [ 58.2],
       [ 73.6],
       [ 76.8],
       [ 69.4],
       [ 75.3],
       [ 79.2],
       [ 80.4],
       [ 73.4],
       [ 67.6],
       [ 62.2],
       [ 64.3],
       [ 76.4],
       [ 55.9],
       [ 80.9],
       [ 74.8],
       [ 78.5],
       [ 56.7],
       [ 55. ],
       [ 81.1],
       [ 74.3],
       [ 67.4],
       [ 69.1],
       [ 46.1],
       [ 81.1],
       [ 81.9],
       [ 69.5],
       [ 59.7],
       [ 74.1],
       [ 60. ],
       [ 71.3],
       [ 76.5],
       [ 75.1],
       [ 57.2],
       [ 68.2],
       [ 79.5],
       [ 78.2],
       [ 76. ],
       [ 68.7],
       [ 75.4],
       [ 52. ],
       [ 49. ]])

	X_fertility = np.array([[ 2.73],
       [ 6.43],
       [ 2.24],
       [ 1.4 ],
       [ 1.96],
       [ 1.41],
       [ 1.99],
       [ 1.89],
       [ 2.38],
       [ 1.83],
       [ 1.42],
       [ 1.82],
       [ 2.91],
       [ 5.27],
       [ 2.51],
       [ 3.48],
       [ 2.86],
       [ 1.9 ],
       [ 1.43],
       [ 6.04],
       [ 6.48],
       [ 3.05],
       [ 5.17],
       [ 1.68],
       [ 6.81],
       [ 1.89],
       [ 2.43],
       [ 5.05],
       [ 5.1 ],
       [ 1.91],
       [ 4.91],
       [ 1.43],
       [ 1.5 ],
       [ 1.89],
       [ 3.76],
       [ 2.73],
       [ 2.95],
       [ 2.32],
       [ 5.31],
       [ 5.16],
       [ 1.62],
       [ 2.74],
       [ 1.85],
       [ 1.97],
       [ 4.28],
       [ 5.8 ],
       [ 1.79],
       [ 1.37],
       [ 4.19],
       [ 1.46],
       [ 4.12],
       [ 5.34],
       [ 5.25],
       [ 2.74],
       [ 3.5 ],
       [ 3.27],
       [ 1.33],
       [ 2.12],
       [ 2.64],
       [ 2.48],
       [ 1.88],
       [ 2.  ],
       [ 2.92],
       [ 1.39],
       [ 2.39],
       [ 1.34],
       [ 2.51],
       [ 4.76],
       [ 1.5 ],
       [ 1.57],
       [ 3.34],
       [ 5.19],
       [ 1.42],
       [ 1.63],
       [ 4.79],
       [ 5.78],
       [ 2.05],
       [ 2.38],
       [ 6.82],
       [ 1.38],
       [ 4.94],
       [ 1.58],
       [ 2.35],
       [ 1.49],
       [ 2.37],
       [ 2.44],
       [ 5.54],
       [ 2.05],
       [ 2.9 ],
       [ 1.77],
       [ 2.12],
       [ 2.72],
       [ 7.59],
       [ 6.02],
       [ 1.96],
       [ 2.89],
       [ 3.58],
       [ 2.61],
       [ 4.07],
       [ 3.06],
       [ 2.58],
       [ 3.26],
       [ 1.33],
       [ 1.36],
       [ 2.2 ],
       [ 1.34],
       [ 1.49],
       [ 5.06],
       [ 5.11],
       [ 1.41],
       [ 5.13],
       [ 1.28],
       [ 1.31],
       [ 1.43],
       [ 7.06],
       [ 2.54],
       [ 1.42],
       [ 2.32],
       [ 4.79],
       [ 2.41],
       [ 3.7 ],
       [ 1.92],
       [ 1.47],
       [ 3.7 ],
       [ 5.54],
       [ 1.48],
       [ 4.88],
       [ 1.8 ],
       [ 2.04],
       [ 2.15],
       [ 6.34],
       [ 1.38],
       [ 1.87],
       [ 2.07],
       [ 2.11],
       [ 2.46],
       [ 1.86],
       [ 5.88],
       [ 3.85]])

	linreg(y, X_fertility)


if __name__ == '__main__':
	main()
