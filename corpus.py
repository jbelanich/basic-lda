class CountMatrix:

	def __init__(self, data=None, numRows=None):
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
		for i in xrange(len(self._data)):
			for key in self._data[i]:
				yield (key, self._data[i][key])

	def nonzero(self):
		for i in xrange(len(self._data)):
			for key in self._data[i]:
				yield (i, key)


class Corpus(CountMatrix):

	def __init__(self, data, vocab):
		self.__vocab = vocab
		self._data = data

	def vocabSize(self):
		return len(self.__vocab)

	def numDocuments(self):
		return len(self._data)

	def print_topic_classifications(self, assignments):
		for index,doc in enumerate(self._data):
			print "Document: ", index
			for word in doc:
				print "\t", self.__vocab[word], ": ", assignments[index,word]

	def print_words_by_topic(self, assignments, topic):
		for word in self.get_words_by_topic(assignments, topic):
			print word

	def get_words_by_topic(self, assignments,topic):
		words = set()
		for index,doc in enumerate(self._data):
			for word in doc:
				if(assignments[index,word] == topic):
					words.add(self.__vocab[word])

		return words