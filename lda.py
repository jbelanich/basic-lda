import numpy as n
import scipy.sparse as sparse
from scipy import *

from corpus import *

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

		self.__vocabCounts = n.zeros([self.vocabSize, self.numTopics])#sparse.dok_matrix((self.vocabSize,self.numTopics))
		self.__documentCounts = n.zeros([self.numDocuments, self.numTopics])#sparse.dok_matrix((self.numDocuments, self.numTopics))
		self.__vocabMarginals = n.zeros([self.numTopics])

		self.initialize()

	def initialize(self):
		"""
			Initializes the assignments and the counts based on said assignments.
			Right now, just gives each word a random topic.
		"""

		#rows, cols = self.__corpus.nonzero()
		for row, col in self.__corpus.nonzero():
			assignment = n.random.randint(0,self.numTopics)
			self.__assignments[row,col] = assignment
			self.__documentCounts[row, assignment] += self.__corpus[row,col]
			self.__vocabCounts[col, assignment] += self.__corpus[row,col]

		self.calculateVocabMarginals()

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
			dist = self.topicDistributionUnorm(row,col)
			test = n.random.uniform(0,dist[len(dist)-1],1)
			newAssignment = self.getAssignmentFromUniform(test,dist)
			if newAssignment != self.__assignments[row,col]:
				self.updateAssignment(row,col,newAssignment)

	def getAssignmentFromUniform(self,unifSample, dist):
		runSum = 0
		for index,prob in enumerate(dist):
			runSum += prob
			if unifSample < runSum:
				return index

	def calculateVocabMarginals(self):
		for k in xrange(self.numTopics):
			counts = []
			for r in xrange(self.__vocabCounts.shape[0]):
				counts.append(self.__vocabCounts[r,k] + self.__beta)

			self.__vocabMarginals[k] = sum(counts)

	def updateAssignment(self, doc, word, newTopic):
		"""
			Update the topic for a given word in a given document, and all relevant counts.
		"""
		oldTopic = int(self.__assignments[doc,word])
		wordCount = self.__corpus[doc,word]
		self.__documentCounts[doc,oldTopic] -= wordCount
		self.__documentCounts[doc,newTopic] += wordCount
		self.__vocabCounts[word,oldTopic] -= wordCount
		self.__vocabCounts[word,newTopic] += wordCount
		self.__vocabMarginals[newTopic] += wordCount
		self.__vocabMarginals[oldTopic] -= wordCount
		self.__assignments[doc,word] = newTopic

	def getExcludedVocabCount(self, word, topic, exclude):
		(m,n) = exclude

		if self.__assignments[m,n] == topic and word == n:
			toSubtract = self.__corpus[m,n]
		else:
			toSubtract = 0

		return self.__vocabCounts[word,topic] - toSubtract

	def getExcludedDocumentCount(self, document, topic, exclude):
		(m,n) = exclude

		if self.__assignments[m,n] == topic and m == document:
			toSubtract = self.__corpus[m,n]
		else:
			toSubtract = 0

		return self.__documentCounts[document,topic] - toSubtract

	def getExcludedVocabNorm(self, topic, exclude):
		(m,n) = exclude

		if self.__assignments[m,n] == topic:
			toSubtract = self.__corpus[m,n]
		else:
			toSubtract = 0

		return self.__vocabMarginals[topic] - toSubtract


	def topicDistribution(self, document, word):
		dist = self.topicDistributionUnorm
		return dist/sum(dist)

	def topicDistributionUnorm(self, document, word):
		dist = []
		for i in xrange(self.numTopics):
			prob = self.getExcludedDocumentCount(document, i, (document,word)) + self.__alpha
			vocabProb = self.getExcludedVocabCount(word, i, (document,word)) + self.__beta
			vocabNorm = self.getExcludedVocabNorm(i, (document,word))
			dist.append(prob * (vocabProb/vocabNorm))

		return dist

	def getAssignments(self):
		return self.__assignments