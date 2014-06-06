import numpy as n
import scipy.sparse as sparse
from scipy import *

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

		self.__documentCounts = n.zeros([self.numDocuments, self.numTopics])
		self.__vocabMarginals = n.zeros([self.numTopics])

		self.__r = n.zeros([self.numTopics])
		self.__rSum = 0

		self.__s = n.zeros([self.numTopics])
		self.__sSum = 0

		self.__qCoeff = n.zeros([self.numTopics])

		self.initialize()

	def initialize(self):
		"""
			Initializes the assignments and the counts based on said assignments.
			Right now, just gives each word a random topic.
		"""

		#temporary storage for the vocabulary counts
		vocabCounts = n.zeros([self.vocabSize, self.numTopics])

		for row, col in self.__corpus.nonzero():
			assignment = n.random.randint(0,self.numTopics)
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

		self.__sSum = n.sum(self.__s)

		self.__vocabCountsCache = WordCountCache(self.__corpus, self.__assignments, self.numTopics, self.__qCoeff)

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
				rSum = self.__rSum
				sSum = self.__sSum
				u = n.random.uniform(0, qSum + self.__sSum + rSum, 1)
				if u <= sSum:
					newTopic = self.sCase(u)
				elif sSum < u <= sSum + rSum:
					newTopic = self.rCase(u-sSum)
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
		#self.__docCountsCache.buildX(d)
		for t in xrange(self.numTopics):
			self.__qCoeff[t] = (self.__alpha + self.__documentCounts[d,t])/self.__vocabMarginals[t]
			if self.__documentCounts[d,t] > 0:
				self.__r[t] = (self.__beta * self.__documentCounts[d,t])/(self.__vocabMarginals[t])
				self.__rSum += self.__r[t]
			else:
				self.__r[t] = 0

	def sCase(self,u):
		sSum = 0
		for t,s in enumerate(self.__s):
			if sSum < u <= (sSum + s):
				return t

			sSum += s

	def rCase(self, u):
		rSum = 0
		for t,r in enumerate(self.__r):
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

		#update r cache
		rOldOldTopic = self.__r[oldTopic]
		rOldNewTopic = self.__r[newTopic]
		self.__r[oldTopic] = (self.__beta * self.__documentCounts[doc,oldTopic])/(self.__vocabMarginals[oldTopic])
		self.__r[newTopic] = (self.__beta * self.__documentCounts[doc,newTopic])/(self.__vocabMarginals[newTopic])
		self.__rSum = self.__rSum + (self.__r[newTopic] - rOldNewTopic) + (self.__r[oldTopic] - rOldOldTopic)

		#update qCoeff cache
		qOldOldTopic = self.__qCoeff[oldTopic]
		qOldNewTopic = self.__qCoeff[newTopic]
		self.__qCoeff[oldTopic] = (self.__alpha + self.__documentCounts[doc,oldTopic])/self.__vocabMarginals[oldTopic]
		self.__qCoeff[newTopic] = (self.__alpha + self.__documentCounts[doc,newTopic])/self.__vocabMarginals[newTopic]

		self.__vocabCountsCache.updateCacheTopics(self.__corpus[doc,word], word, oldTopic, newTopic)

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

	def getTopics(self):
		"""
		Returns a list of the topics in this topic model. A topic is a probability
		distribution over words, represented as a numpy array.
		"""
		topics = []
		for i in xrange(self.numTopics):
			topic = (self.__vocabCounts[:,i] + self.__beta)/float(self.__vocabMarginals[i])
			topics.append(topic)

		return topics