#!/usr/bin/python3

import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


def main():
	''' Main function for testing purposes'''

	df = pd.read_csv('titanic.csv')

	# Drop the target column
	predictors = df.drop(['survived'], axis=1)
	n_cols = len(predictors.columns)

	# Convert the target to categorical: target
	target = to_categorical(df.survived)

	# Impute missing data
	mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
	mean_imputer = mean_imputer.fit(predictors)
	imputed_df = mean_imputer.transform(predictors.values)

	# Send data to classifier model
	classification_model(imputed_df, target, n_cols)


def classification_model(predictors, target, n_cols):
	''' Create a classifciation model'''

	# Set up the model
	model = Sequential()

	# Add the first layer
	model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

	# Add the output layer
	model.add(Dense(2, activation='softmax'))

	# Compile the model
	model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

	# Fit the model
	model.fit(predictors, target, epochs=10)

	# Save the model
	model.save('model_file.h5')


if __name__ == '__main__':
	main()
