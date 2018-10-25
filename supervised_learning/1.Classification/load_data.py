#!/usr/bin/python3

import pandas as pd
import numpy as np


def main():
	'''main function for testing'''

	dt_file = 'fertility.csv'
	fertility(dt_file)


def votes(data_file):
	'''loads votes.csv data file'''

	df = pd.read_csv(data_file)

	# Change the 'yes' and 'no' votes to 1 or 0
	df.replace(('y', 'n'), (1, 0), inplace=True)

	# Change the '?' to nan then drop them
	df[df == '?'] = np.nan
	df = df.dropna()

	# Show the dataframe if testing
	if __name__ == '__main__':
		print(df)

	return df


def fertility(data_file):
	''' loads fertility csv file'''

	df = pd.read_csv(data_file)
	df = df.dropna()
	print(df)
	

	return df


if __name__ == '__main__':
	main()
