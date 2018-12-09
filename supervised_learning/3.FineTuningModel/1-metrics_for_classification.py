#!/usr/bin/python3

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
	'''Evaluates the performance of a binary classifier by computing a confusion matrix and generating a classification report
	   Accounts for imbalances in data (e.g., 99% email-spam example)

       Confusion matrix:
                             Predicted          Predicted
                             Spam Email         Real Email

         Actual: Spam Email  True Positive      False Negative

         Actual: Real Email  False Positive     True Negative

	METRICS:
	Accuracy = (tp + tn) / (tp + tn + fp + fn)   

	Precision = tp / (tp + fp) (high=not many real emails predicted as spam)

	Recall = tp / (tp + fn) aka sensitivity, hit rate, or true positive rate (high=predicted most spam emails corrctly)

	F1 score = 2 * (precision * recall) / (precision + recall)
    '''
	

	file_name = 'diabetes.csv'
	df = pd.read_csv(file_name)

	y = np.array(df['Outcome'])
	X = np.array(df.drop('Outcome', axis=1))

	# Create training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	# Instantiate a k-NN classifier: knn
	knn = KNeighborsClassifier(n_neighbors=6)

	# Fit the classifier to the training data
	knn.fit(X_train, y_train)

	# Predict the labels of the test data: y_pred
	y_pred = knn.predict(X_test)

	# Generate the confusion matrix and classification report
	print('CONFUSION MATRIX')
	print(confusion_matrix(y_test, y_pred))
	print('')
	print('CLASSIFICATION REPORT')
	print(classification_report(y_test, y_pred))


if __name__ == '__main__':
	main()
