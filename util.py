import cProfile as profile
from lda import *
from data_generation import *

def timeExp(corpus=None):
	if corpus is None:
		corpus = dailyKosCountMatrix(1000)

	ldaTest = LDAModel(corpus, numTopics=100)
	p = profile.Profile()
	p.enable()
	ldaTest.fastGibbs(1)
	p.disable()
	p.print_stats(sort='cumtime')

def nipsTimeExp(numDocs=100):
	corpus = filesToCorpus('./nipsvocab.txt', './docword.nips.txt',numDocs=numDocs)
	ldaTest = LDAModel(corpus, numTopics=100)
	p = profile.Profile()
	p.enable()
	ldaTest.fastGibbs(1)
	p.disable()
	p.print_stats(sort='cumtime')

def baseExp(numDocuments, topics, iterations):
	corpus = dailyKosCountMatrix(numDocuments)
	return fullExp(corpus, topics, iterations)

def fullExp(corpus, topics, iterations):
	ldaTest = LDAModel(corpus, numTopics=topics)
	ldaTest.fastGibbs(iterations)
	assignments = ldaTest.getAssignments()
	return corpus, assignments, ldaTest

def nipsExp(numDocuments, topics, iterations):
	corpus = filesToCorpus('./nipsvocab.txt', './docword.nips.txt',numDocs=numDocuments)
	return fullExp(corpus, topics, iterations)