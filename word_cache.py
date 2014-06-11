from corpus import *
from bisect import *

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

		for count,topic in countList:
			val = self.calculateCacheValue(count,topic)
			self._xCache.append((topic, val))
			self._xSum += val

	def slice(self, topic):
		"""
		Generator that gives you an array of length len(self._data) where
		arr[i] gives you count in (topic,count) in self._data[i].
		"""
		for countList in self._data:
			#find the corresponding entry
			count = 0
			for c,t in countList:
				if topic == t:
					count = c
					break

			yield count

	def removeCacheTopics(self, count, countIndex, oldTopic):
		countList = self._data[countIndex]

		oldIndex = None
		for index,(score,topic) in enumerate(countList):
			if topic == oldTopic:
				oldIndex = index
				break

		updateCount,_ = countList.pop(oldIndex)
		updateCount -= count
		if updateCount > 0:
			insort(countList, (updateCount,oldTopic))

	def addCacheTopics(self, count, countIndex, newTopic):
		countList = self._data[countIndex]

		newIndex = None
		for index,(score,topic) in enumerate(countList):
			if topic == newTopic:
				newIndex = index
				break

		if newIndex is None:
			updateCount = count
		else:
			updateCount,_ = countList.pop(newIndex)
			updateCount += count

		insort(countList, (updateCount, newTopic))

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
			self._data[w].append((vocabCounts[w,t],t))

		#sort each bucket in descending order
		for w,wordList in enumerate(self._data):
			self._data[w] = sorted(wordList, reverse=True)

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

		docCounts = CountMatrix(numRows=self._corpus.numDocuments())

		for row,col in self._corpus.nonzero():
			assignment = self._assignments[row,col]
			docCounts[row,assignment] += self._corpus[row,col]

		for d,t in docCounts.nonzero():
			self._data[d].append((docCounts[d,t],t))

		for d,docList in enumerate(self._data):
			self._data[d] = sorted(docList, reverse=True)

	def calculateCacheValue(self, count, topic):
		return (count * self._beta) / self._vocabMarginals[topic]