import random

from corpus import *
from util import *
from word_cache import *

class LDAModel:

	def __init__(self, corpus, numTopics = 10, alpha=.1, beta=.1):
		"""
			Initializes a LDAModel based on the given corpus. Alpha and
			Beta represent the hyperparameters of the model. If none are
			provided, we generate them.
		"""
		self.numDocuments = corpus.numDocuments()
		self.vocabSize = corpus.vocabSize()
		self.numTopics = numTopics

		self.__alpha = alpha
		self.__beta = beta

		self.__corpus = corpus
		self.__assignments = CountMatrix(numRows=corpus.numRows())

		self.__documentCounts = CountMatrix(numRows=corpus.numRows())#n.zeros([self.numDocuments, self.numTopics])
		self.__vocabMarginals = [0 for _ in xrange(self.numTopics)]#n.zeros([self.numTopics])

		self.__s = [0 for _ in xrange(self.numTopics)]#n.zeros([self.numTopics])
		self.__sSum = 0

		self.__qCoeff = [0 for _ in xrange(self.numTopics)]#n.zeros([self.numTopics])

		self.initialize()

	def initialize(self):
		"""
			Initializes the assignments and the counts based on said assignments.
			Right now, just gives each word a random topic.
		"""

		#temporary storage for the vocabulary counts
		vocabCounts = CountMatrix(numRows=self.vocabSize)#n.zeros([self.vocabSize, self.numTopics])

		for row, col in self.__corpus.nonzero():
			assignment = random.randint(0,self.numTopics-1)
			self.__assignments[row,col] = assignment
			self.__documentCounts[row, assignment] += self.__corpus[row,col]
			vocabCounts[col, assignment] += self.__corpus[row,col]

		#building the vocab marginals
		for k in xrange(self.numTopics):
			counts = []
			for r in xrange(self.__corpus.vocabSize()):
				counts.append(vocabCounts[r,k] + self.__beta)

			self.__vocabMarginals[k] = sum(counts)

		#initialize the s cache
		for t in xrange(self.numTopics):
			self.__s[t] = (self.__alpha * self.__beta)/(self.__vocabMarginals[t])

		self.__sSum = sum(self.__s)

		self.__vocabCountsCache = WordCountCache(self.__corpus, self.__assignments, self.numTopics, self.__qCoeff)
		self.__documentCountsCache = DocumentCountCache(self.__corpus, self.__assignments, self.numTopics, self.__vocabMarginals, self.__beta)

	def fastGibbs(self, iterations):
		for i in xrange(iterations):
			print "Begin Fast Iteration", i
			self.fastGibbsStep()

	def fastGibbsStep(self):
		for d in xrange(self.__corpus.numDocuments()):
			self.buildCachedValuesForDocument(d)
			for w in self.__corpus.columnsInRow(d):
				self.__vocabCountsCache.buildX(w)
				qSum = self.__vocabCountsCache.getXSum(w)
				rSum = self.__documentCountsCache.getXSum(d)
				sSum = self.__sSum
				u = random.uniform(0, qSum + rSum + sSum)#n.random.uniform(0, qSum + self.__sSum + rSum, 1)
				if u <= sSum:
					newTopic = self.sCase(u)
				elif sSum < u <= sSum + rSum:
					newTopic = self.rCase(u-sSum, d)
				elif sSum + rSum < u <= sSum + rSum + qSum:
					newTopic = self.qCase(u-sSum-rSum,w)
				else:
					assert(False)

				if newTopic != self.__assignments[d,w]:
					self.updateCachedValues(d,w,newTopic)

	def buildCachedValuesForDocument(self,d):
		"""
		Populates r, rSum, and qCoeff
		"""
		self.__rSum = 0
		self.__documentCountsCache.buildX(d)
		for t in xrange(self.numTopics):
			self.__qCoeff[t] = (self.__alpha + self.__documentCounts[d,t])/self.__vocabMarginals[t]

	def sCase(self,u):
		sSum = 0
		for t,s in enumerate(self.__s):
			if sSum < u <= (sSum + s):
				return t

			sSum += s

	def rCase(self, u, d):
		rSum = 0
		for t,r in self.__documentCountsCache.getX(d):
			if rSum < u <= (rSum + r):
				return t

			rSum += r

	def qCase(self, u, w):
		qSum = 0
		for t,q in self.__vocabCountsCache.getX(w):
			if qSum < u <= (qSum + q):
				return t

			qSum += q

	def updateCachedValues(self,doc,word,newTopic):
		wordCount = self.__corpus[doc,word]
		oldTopic = self.__assignments[doc,word]

		#remove old assignments
		self.__documentCounts[doc,oldTopic] -= wordCount
		self.__vocabMarginals[oldTopic] -= wordCount

		#update new assignments
		self.__documentCounts[doc,newTopic] += wordCount
		self.__vocabMarginals[newTopic] += wordCount

		self.__assignments[doc,word] = newTopic

		#update s cache
		sOldOldTopic = self.__s[oldTopic]
		sOldNewTopic = self.__s[newTopic]
		self.__s[oldTopic] = (self.__alpha * self.__beta)/(self.__vocabMarginals[oldTopic])
		self.__s[newTopic] = (self.__alpha * self.__beta)/(self.__vocabMarginals[newTopic])
		self.__sSum = self.__sSum + (self.__s[oldTopic] - sOldOldTopic) + (self.__s[newTopic] - sOldNewTopic)

		#update qCoeff cache
		qOldOldTopic = self.__qCoeff[oldTopic]
		qOldNewTopic = self.__qCoeff[newTopic]
		self.__qCoeff[oldTopic] = (self.__alpha + self.__documentCounts[doc,oldTopic])/self.__vocabMarginals[oldTopic]
		self.__qCoeff[newTopic] = (self.__alpha + self.__documentCounts[doc,newTopic])/self.__vocabMarginals[newTopic]

		self.__vocabCountsCache.updateCacheTopics(self.__corpus[doc,word], word, oldTopic, newTopic)
		self.__documentCountsCache.updateCacheTopics(self.__corpus[doc,word], doc, oldTopic, newTopic)

	def getAssignments(self):
		return self.__assignments

	def getTopicProbabilitiesForDocument(self):
		"""
		Returns a list where the ith' element is the topic probability distribution for the i'th
		document.
		"""
		latentTopics = []
		for i in xrange(self.numDocuments):
			topicDist = (self.__documentCounts[i,:] + self.__alpha) / \
			(float(sum(self.__documentCounts[i,:])) + (self.numTopics * self.__alpha))
			latentTopics.append(topicDist)

		return latentTopics

##need to fix this

	# def getTopics(self):
	# 	"""
	# 	Returns a list of the topics in this topic model. A topic is a probability
	# 	distribution over words, represented as a numpy array.
	# 	"""
	# 	topics = []
	# 	for i in xrange(self.numTopics):
	# 		topic = (self.__vocabCounts[:,i] + self.__beta)/float(self.__vocabMarginals[i])
	# 		topics.append(topic)

	# 	return topics