#!/usr/bin/python3

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
	''' Create a tf-idf word frequency array'''

	documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']

	# Create a TfidfVectorizer: tfidf
	tfidf = TfidfVectorizer()

	# Apply fit_transform to document: csr_mat
	csr_mat = tfidf.fit_transform(documents)

	# Print result of toarray() method
	print(csr_mat.toarray())

	# Get the words: words
	words = tfidf.get_feature_names()

	# Print words
	print(words)


if __name__ == '__main__':
	main()
