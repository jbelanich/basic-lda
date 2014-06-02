import numpy as n
import scipy.sparse as sparse
from scipy import *

class LDAModel:

	def __init__(self, corpus, numTopics = 10, alpha=None, beta=None):
		"""
			Initializes a LDAModel based on the given corpus. Alpha and
			Beta represent the hyperparameters of the model. If none are
			provided, we generate them.
		"""
		self.numDocuments = corpus.shape[0]
		self.vocabSize = corpus.shape[1]
		self.numTopics = numTopics

		if not alpha:
			self.generateAlpha()
		else:
			self.__alpha = alpha

		if not beta:
			self.generateBeta()
		else:
			self.__beta = beta

		self.__corpus = corpus
		self.__assignments = sparse.lil_matrix(corpus.shape, dtype=int64)

		self.__vocabCounts = sparse.lil_matrix((self.vocabSize,self.numTopics))
		self.__documentCounts = sparse.lil_matrix((self.numDocuments, self.numTopics))
		self.__vocabMarginals = n.zeros([self.numTopics])

		self.initialize()

	def initialize(self):
		"""
			Initializes the assignments and the counts based on said assignments.
			Right now, just gives each word a random topic.
		"""

		rows, cols = self.__corpus.nonzero()
		for row, col in zip(rows,cols):
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
			print "============================="
			self.gibbsStep()

	def gibbsStep(self):
		"""
			Refresh all topic assignments.
		"""
		rows, cols = self.__corpus.nonzero()
		for row, col in zip(rows,cols):
			#construct probability dist over topic assignments
			dist = self.topicDistribution(row,col)
			print dist
			newAssignment = n.random.choice(self.numTopics, p=dist)
			if newAssignment != self.__assignments[row,col]:
				self.updateAssignment(row,col,newAssignment)

	def calculateVocabMarginals(self):
		for k in xrange(self.numTopics):
			counts = []
			rows, cols = self.__vocabCounts[:,k].nonzero()
			for r in set(rows):
				counts.append(self.__vocabCounts[r,k] + self.__beta[r])

			self.__vocabMarginals[k] = sum(counts)

	def updateAssignment(self, doc, word, newTopic):
		"""
			Update the topic for a given word in a given document, and all relevant counts.
		"""
		oldTopic = self.__assignments[doc,word]
		self.__documentCounts[doc,oldTopic] -= self.__corpus[doc,word]
		self.__documentCounts[doc,newTopic] += self.__corpus[doc,word]
		self.__vocabCounts[word,oldTopic] -= self.__corpus[doc,word]
		self.__vocabCounts[word,newTopic] += self.__corpus[doc,word]
		self.__vocabMarginals[newTopic] += self.__corpus[doc,word]
		self.__vocabMarginals[oldTopic] -= self.__corpus[doc,word]
		self.__assignments[doc,word] = newTopic

	def getExcludedVocabCount(self, word, topic, exclude):
		(m,n) = exclude

		#toSubtract = self.__corpus[m,n] if self.__assignments[m,n] == topic else 0
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
		#first calculate vocabulary normalization
		dist = []
		for i in xrange(self.numTopics):
			prob = self.getExcludedDocumentCount(document, i, (document,word)) + self.__alpha[i]
			vocabProb = self.getExcludedVocabCount(word, i, (document,word)) + self.__beta[word]
			vocabNorm = self.getExcludedVocabNorm(i, (document,word))
			dist.append(prob * (vocabProb/vocabNorm))

		return dist/sum(dist)

	def getAssignments(self):
		return self.__assignments

	def generateAlpha(self):
		self.__alpha = n.ones(self.numTopics)

	def generateBeta(self):
		self.__beta = n.ones(self.vocabSize)