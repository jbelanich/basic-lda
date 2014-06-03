import cProfile as profile
from lda import *
from data_generation import *

def runExp(corpus):
	ldaTest = LDAModel(corpus, numTopics=50)
	p = profile.Profile()
	p.enable()
	ldaTest.gibbs(1)
	p.disable()
	p.print_stats(sort='cumtime')

def baseExp(numDocuments, topics, iterations):
	corpus = dailyKosCountMatrix(numDocuments)
	ldaTest = LDAModel(corpus, numTopics=topics)
	ldaTest.gibbs(iterations)
	assignments = ldaTest.getAssignments()
	return corpus,assignments