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
			generateAlpha()
		else
			self.__alpha = alpha

		if not beta:
			generateBeta()
		else
			self.__beta = beta

		self.__corpus = corpus
		self.__assignments = sparse.csc_matrix(corpus.shape, dtype=int64)

		self.numDocuments = corpus.shape[0]
		self.vocabSize = corpus.shape[1]
		self.numTopics = numTopics

		self.__vocabCounts = sparse.csc_matrix((self.vocabSize,self.numTopics))
		self.__documentCounts = sparse.

		initialize()

	def initialize(self):
		"""
			Initializes the assignments and the counts based on said assignments.
			Right now, just gives each word a random topic.
		"""

	def generateAlpha(self):
		pass

	def generateBeta(self):
		pass