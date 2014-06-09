from corpus import *

class CountCache(object):

	def __init__(self, corpus, assignments, numTopics):
		self._data = []
		self._corpus = corpus
		self._assignments = assignments
		self._numTopics = numTopics
		self._xCache = []
		self._xSum = 0

		self.build()

	def build(self):
		pass

	def calculateCacheValue(self, count):
		pass

	def getX(self, countIndex):
		for e in self._xCache:
			yield e

	def getXSum(self, countIndex):
		return self._xSum

	def buildX(self, countIndex):
		countList = self._data[countIndex]
		self._xCache = []
		self._xSum = 0

		for topic,count in countList:
			val = self.calculateCacheValue(count,topic)
			self._xCache.append((topic, val))
			self._xSum += val

	def updateCacheTopics(self, count, countIndex, oldTopic, newTopic):
		countList = self._data[countIndex]

		oldIndex = None
		newIndex = None

		for index,(topic,_) in enumerate(countList):
			if topic == oldTopic:
				oldIndex = index
			elif topic == newTopic:
				newIndex = index

		topic,oldCount = countList[oldIndex]
		updateCount = oldCount - count
		countList[oldIndex] = (topic,updateCount)

		#the count for newtopic is 0 and so isn't in the array
		if newIndex is None:
			countList.append((newTopic, count))
		else:
			topic,newCount = countList[newIndex]
			countList[newIndex] = (topic, newCount + count)

		if updateCount == 0:
			countList.pop(oldIndex)

		self._data[countIndex] = sorted(countList, key=lambda x: x[1], reverse=True)

class WordCountCache(CountCache):

	def __init__(self, corpus, assignments, numTopics, qCoeff):
		self._qCoeffs = qCoeff
		CountCache.__init__(self, corpus, assignments, numTopics)

	def build(self):
		for w in xrange(self._corpus.vocabSize()):
			self._data.append([])

		vocabCounts = CountMatrix(numRows=self._corpus.vocabSize())

		for row, col in self._corpus.nonzero():
			assignment = self._assignments[row,col]
			vocabCounts[col, assignment] += self._corpus[row,col]

		for w,t in vocabCounts.nonzero():
			self._data[w].append((t, vocabCounts[w,t]))

		#sort each bucket in descending order
		for w,wordList in enumerate(self._data):
			self._data[w] = sorted(wordList, key=lambda x: x[1], reverse=True)

	def calculateCacheValue(self, count, topic):
		return count * self._qCoeffs[topic]

class DocumentCountCache(CountCache):

	def __init__(self, corpus, assignments, numTopics, vocabMarginals, beta):
		self._beta = beta
		self._vocabMarginals = vocabMarginals
		CountCache.__init__(self, corpus, assignments, numTopics)

	def build(self):
		for d in xrange(self._corpus.numDocuments()):
			self._data.append([])

		docCounts = CountMatrix(numRows=self._corpus.numDocuments())#n.zeros([self._corpus.numDocuments(), self._numTopics])

		for row,col in self._corpus.nonzero():
			assignment = self._assignments[row,col]
			docCounts[row,assignment] += self._corpus[row,col]

		for d,t in docCounts.nonzero():
			self._data[d].append((t, docCounts[d,t]))

		for d,docList in enumerate(self._data):
			self._data[d] = sorted(docList, key=lambda x: x[1], reverse=True)

	def calculateCacheValue(self, count, topic):
		return (count * self._beta) / self._vocabMarginals[topic]