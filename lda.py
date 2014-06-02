import numpy as n
import scipy.sparse as sparse
from scipy import *

numTopics = 20

class Document:

	def __init__(self):
		self.raw = ""
		self.words = []


class LDAModel:

	def __init__(self, corpus, alpha=None, beta=None):
		"""
			Initializes a LDAModel based on the given corpus. Alpha and
			Beta represent the hyperparameters of the model. If none are
			provided, we generate them.
		"""
		if not alpha:
			self.generateAlpha()
		else:
			self.__alpha = alpha

		if not beta:
			self.generateBeta()
		else:
			self.__beta = beta

		self.__corpus = corpus
		self.__assignments = sparse.csc_matrix(corpus.shape, dtype=int64)

		self.numDocuments = corpus.shape[0]
		self.vocabSize = corpus.shape[1]
		self.numTopics = numTopics

		self.__vocabCounts = sparse.csc_matrix((self.vocabSize,self.numTopics))
		self.__documentCounts = sparse.csc_matrix((self.numDocuments, self.numTopics))

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

	def gibbs(self, iterations):
		"""
			Runs gibbs sampling step for the provided number of iterations.
		"""

		pass

	def generateAlpha(self):
		pass

	def generateBeta(self):
		pass