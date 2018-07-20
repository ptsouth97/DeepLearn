#!/usr/bin/python3

import pandas as pd

def main():
	'''load file votes.csv'''
	
	data_file = 'votes.csv'
	df = pd.read_csv(data_file)
	print(df)


if __name__ == '__main__':
	main()
