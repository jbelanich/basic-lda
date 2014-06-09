import cProfile as profile
from lda import *
from data_generation import *

def timeExp(corpus=None):
	if corpus is None:
		corpus = dailyKosCorpus(100)

	ldaTest = LDAModel(corpus, numTopics=50)
	p = profile.Profile()
	p.enable()
	ldaTest.fastGibbs(1)
	p.disable()
	p.print_stats(sort='cumtime')

def nipsTimeExp(numDocs=100):
	corpus = nipsCorpus()
	ldaTest = LDAModel(corpus, numTopics=100)
	p = profile.Profile()
	p.enable()
	ldaTest.fastGibbs(1)
	p.disable()
	p.print_stats(sort='cumtime')

def baseExp(numDocuments, topics, iterations):
	corpus = dailyKosCorpus(numDocuments)
	return fullExp(corpus, topics, iterations)

def fullExp(corpus, topics, iterations):
	ldaTest = LDAModel(corpus, numTopics=topics)
	ldaTest.fastGibbs(iterations)
	assignments = ldaTest.getAssignments()
	return corpus, assignments, ldaTest

def nipsExp(numDocuments, topics, iterations):
	corpus = nipsCorpus(numDocuments)
	return fullExp(corpus, topics, iterations)