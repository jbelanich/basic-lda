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

		self.__vocabCounts = n.zeros([self.vocabSize, self.numTopics])
		self.__documentCounts = n.zeros([self.numDocuments, self.numTopics])
		self.__vocabMarginals = n.zeros([self.numTopics])

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
			dist = self.topicDistributionUnorm(row,col)
			newAssignment = self.selectTopic(dist)
			self.updateAssignment(row,col,newAssignment)

	def selectTopic(self, dist):
		"""
		Selects a topic from the provided *unnormalized* probability distribution dist.
		"""
		u = n.random.uniform(0,sum(dist),1)

		runSum = 0
		for index,prob in enumerate(dist):
			runSum += prob
			if u < runSum:
				return index

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

	def getAssignments(self):
		return self.__assignments