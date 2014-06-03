class CountMatrix:

	def __init__(self, data=None, numDocs=None):
		if data:
			self._data = data
		else:
			self._data = []
			for i in xrange(numDocs):
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

	def getVocabSize(self):
		return len(self.__vocab)

	def getNumDocuments(self):
		return self.__wordCounts.numRows()