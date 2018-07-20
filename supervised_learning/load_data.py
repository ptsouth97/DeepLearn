#!/usr/bin/python3

import pandas as pd
import numpy as np


def main():
	'''load file votes.csv'''
	
	data_file = 'votes.csv'
	df = pd.read_csv(data_file)
	df.replace(('y', 'n'), (1, 0), inplace=True)
	df[df == '?'] = np.nan
	df = df.dropna()

	print(df)

	return df


if __name__ == '__main__':
	main()
