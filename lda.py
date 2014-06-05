import numpy as n
import scipy.sparse as sparse
from scipy import *

from corpus import *
from util import *

class LDAModel:

	def __init__(self, corpus, numTopics = 10, alpha=1, beta=1):
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

		self.__vocabCounts = n.zeros([self.vocabSize, self.numTopics])
		self.__documentCounts = n.zeros([self.numDocuments, self.numTopics])
		self.__vocabMarginals = n.zeros([self.numTopics])

		self.__r = n.zeros([self.numTopics])
		self.__rSum = 0

		self.__s = n.zeros([self.numTopics])
		self.__sSum = 0

		self.__qCoeff = n.zeros([self.numTopics])
		self.__q = n.zeros([self.numTopics])
		self.__qSum = 0

		self.initialize()

	def initialize(self):
		"""
			Initializes the assignments and the counts based on said assignments.
			Right now, just gives each word a random topic.
		"""
		for row, col in self.__corpus.nonzero():
			assignment = n.random.randint(0,self.numTopics)
			self.__assignments[row,col] = assignment
			self.__documentCounts[row, assignment] += self.__corpus[row,col]
			self.__vocabCounts[col, assignment] += self.__corpus[row,col]

		self.calculateVocabMarginals()
		self.calculateS()

	def calculateS(self):
		for t in xrange(self.numTopics):
			self.__s[t] = (self.__alpha * self.__beta)/(self.__vocabMarginals[t])

		self.__sSum = n.sum(self.__s)

	def fastGibbs(self, iterations):
		for i in xrange(iterations):
			print "Begin Fast Iteration", i
			self.fastGibbsStep()

	def buildCachedValuesForDocument(self,d):
		"""
		Populates r, rSum, and qCoeff
		"""
		self.__rSum = 0
		for t in xrange(self.numTopics):
			self.__qCoeff[t] = (self.__alpha + self.__documentCounts[d,t])/self.__vocabMarginals[t]
			if self.__documentCounts[d,t] > 0:
				self.__r[t] = (self.__beta * self.__documentCounts[d,t])/(self.__vocabMarginals[t])
				self.__rSum += self.__r[t]
			else:
				self.__r[t] = 0

	def buildCachedValuesForWord(self,d,w):
		"""
		Populates q and qSum
		"""
		self.__qSum = 0
		for t in xrange(self.numTopics):
			if self.__vocabCounts[w,t] > 0:
				self.__q[t] = self.__qCoeff[t] * self.__vocabCounts[w,t]
				self.__qSum += self.__q[t]
			else:
				self.__q[t] = 0

	def fastGibbsStep(self):
		for d in xrange(self.__corpus.numDocuments()):
			self.buildCachedValuesForDocument(d)
			for w in self.__corpus.columnsInRow(d):
				self.buildCachedValuesForWord(d,w)
				u = n.random.uniform(0, self.__qSum + self.__sSum + self.__rSum, 1)
				if u <= self.__sSum:
					newTopic = self.sCase(u)
				elif self.__sSum < u <= self.__sSum + self.__rSum:
					newTopic = self.rCase(u)
				elif self.__sSum + self.__rSum < u <= self.__sSum + self.__rSum + self.__qSum:
					newTopic = self.qCase(u)
				else:
					assert(False)

				self.updateCachedValues(d,w,newTopic)

	def sCase(self,u):
		sSum = 0
		for t in xrange(self.numTopics):
			if sSum < u <= (sSum + self.__s[t]):
				return t

			sSum += self.__s[t]

	def rCase(self, u):
		rSum = self.__sSum
		for t in xrange(self.numTopics):
			if rSum < u <= (rSum + self.__r[t]):
				return t

			rSum += self.__r[t]

	def qCase(self, u):
		qSum = self.__rSum + self.__sSum
		for t in xrange(self.numTopics):
			if qSum < u <= (qSum + self.__q[t]):
				return t

			qSum += self.__q[t]

	def gibbs(self, iterations):
		"""
			Runs gibbs sampling step for the provided number of iterations.
		"""

		for i in xrange(iterations):
			print "BEGIN ITERATION", i
			self.gibbsStep()

	def gibbsStep(self):
		"""
			Refresh all topic assignments.
		"""
		for row, col in self.__corpus.nonzero():
			#construct probability dist over topic assignments
			self.removeAssignmentsForWord(row,col)
			dist = self.topicDistributionUnormSum(row,col)
			newAssignment = self.selectTopic(dist)#self.selectTopic(dist)
			self.updateAssignment(row,col,newAssignment)

	def selectTopic(self, dist):
		"""
		Selects a topic from the provided *unnormalized* probability distribution dist.
		"""
		u = n.random.uniform(0,dist[-1],1)

		#return binary_search(u, dist)
		for index,prob in enumerate(dist):
			if u < prob:
				return index

	def selectTopicOld(self, dist):
		return n.random.choice(self.numTopics, p=dist/n.sum(dist))

	def calculateVocabMarginals(self):
		for k in xrange(self.numTopics):
			counts = []
			for r in xrange(self.__vocabCounts.shape[0]):
				counts.append(self.__vocabCounts[r,k] + self.__beta)

			self.__vocabMarginals[k] = sum(counts)

	def removeAssignmentsForWord(self, doc, word):
		topic = self.__assignments[doc,word]
		wordCount = self.__corpus[doc,word]
		self.__documentCounts[doc,topic] -= wordCount
		self.__vocabCounts[word,topic] -= wordCount
		self.__vocabMarginals[topic] -= wordCount

	def updateAssignment(self, doc, word, newTopic):
		"""
			Update the topic for a given word in a given document, and all relevant counts.
		"""
		wordCount = self.__corpus[doc,word]
		self.__documentCounts[doc,newTopic] += wordCount
		self.__vocabCounts[word,newTopic] += wordCount
		self.__vocabMarginals[newTopic] += wordCount
		self.__assignments[doc,word] = newTopic

	def updateCachedValues(self,doc,word,newTopic):
		wordCount = self.__corpus[doc,word]
		oldTopic = self.__assignments[doc,word]

		#remove old assignments
		self.__documentCounts[doc,oldTopic] -= wordCount
		self.__vocabCounts[word,oldTopic] -= wordCount
		self.__vocabMarginals[oldTopic] -= wordCount

		#update new assignments
		self.__documentCounts[doc,newTopic] += wordCount
		self.__vocabCounts[word,newTopic] += wordCount
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

	def topicDistribution(self, document, word):
		dist = self.topicDistributionUnorm
		return dist/sum(dist)

	def topicDistributionUnorm(self, document, word):
		"""
		Returns the *unnormalized* topic distribution for the given word in the given document.
		"""
		dist = []
		for i in xrange(self.numTopics):
			prob = self.__documentCounts[document,i] + self.__alpha
			vocabProb = self.__vocabCounts[word,i] + self.__beta
			vocabNorm = self.__vocabMarginals[i] #don't need beta here, its already baked in
			dist.append(prob * (vocabProb/vocabNorm))

		return dist

	def topicDistributionUnormSum(self, document, word):
		dist = []
		last = 0
		for i in xrange(self.numTopics):
			prob = self.__documentCounts[document,i] + self.__alpha
			vocabProb = self.__vocabCounts[word,i] + self.__beta
			vocabNorm = self.__vocabMarginals[i] #don't need beta here, its already baked in
			curr = prob*(vocabProb/vocabNorm)
			dist.append(last + curr)
			last = curr

		return dist

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