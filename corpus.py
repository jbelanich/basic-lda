import numpy as np

class CountMatrix:
	"""
		A wrapper class for a list of dicts.

		For example, if obj is a CountMatrix, then obj[i,j] gets
		the thing mapped to j in the i'th dictionary.
	"""

	def __init__(self, **kwargs):
		"""
		Allows for two possiblities. Either you provide a list of dicts
		to use as the data for this CountMatrix, or you provide a integer
		giving the number of dicts in this CountMatrix.
		"""
		data = kwargs.get('data', None)
		numRows = kwargs.get('numRows', None)

		assert(data or numRows)

		if data:
			self._data = data
		else:
			self._data = []
			for i in xrange(numRows):
				self._data.append({})

	def __getitem__(self, pos):
		row,col = pos
		return self._data[row].get(col, 0)

	def __setitem__(self, pos, value):
		row,col = pos
		self._data[row][col] = value

	def numRows(self):
		return len(self._data)

	def elements(self):
		"""
		Iterates over the elements of this CountMatrix. That is,
		returns tuples of the form (wordIndex, wordCount).
		"""
		for row in self._data:
			for key in row:
				yield (key, row[key])

	def nonzero(self):
		"""
		Iterates over all of the indicies of this CountMatrix. This is,
		returns tuples of the form (documentIndex,wordIndex).
		"""
		for i,row in enumerate(self._data):
			for key in row:
				yield (i, key)

	def toCorpus(self, vocab):
		"""
		Returns a Corpus for this count matrix with the provided vocabulary.
		"""
		return Corpus(self._data, vocab)

class Corpus(CountMatrix):
	"""
		A Corpus is a special kind of CountMatrix, where corpus[i,j] gives
		us the number of occurances of the j'th word in the i'th document.

		A Corpus also has a vocabulary modeled as a list of strings. The string representation
		of the i'th word is stored at vocab[i].
	"""
	def __init__(self, data, vocab):
		self.__vocab = vocab
		self._data = data

	def vocabSize(self):
		return len(self.__vocab)

	def numDocuments(self):
		return len(self._data)

	def print_topic_classifications(self, assignments):
		"""
		Prints all words in all documents, with their topic assignments.
		"""
		for index,doc in enumerate(self._data):
			print "Document: ", index
			for word in doc:
				print "\t", self.__vocab[word], ": ", assignments[index,word]

	def print_words_by_topic(self, assignments, topic):
		"""
		Prints the set of words in the corpus that are mapped to the provided topic.
		"""
		for word in self.get_words_by_topic(assignments, topic):
			print word

	def get_words_by_topic(self, assignments,topic):
		"""
		Returns the set of words in the corpus that are mapped to the provided topic.
		"""
		words = set()
		for index,doc in enumerate(self._data):
			for word in doc:
				if(assignments[index,word] == topic):
					words.add(self.__vocab[word])

		return words

	def get_top_words_for_topic(self, topic, n=5):
		wordInd = np.argsort(topic)
		words = []
		for i in wordInd:
			words.append(self.__vocab[i])
		return ' '.join(words[-n:])

	def getVocab(self):
		return self.__vocab