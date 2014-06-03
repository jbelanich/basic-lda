import cProfile as profile
from lda import *

def runExp(corpus):
	ldaTest = LDAModel(corpus, numTopics=20)
	p = profile.Profile()
	p.enable()
	ldaTest.gibbs(1)
	p.disable()
	p.print_stats(sort='cumtime')