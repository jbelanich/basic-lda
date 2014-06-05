import cProfile as profile
from lda import *
from data_generation import *

def timeExp(corpus):
	ldaTest = LDAModel(corpus, numTopics=100)
	p = profile.Profile()
	p.enable()
	ldaTest.fastGibbs(10)
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

def binary_search(value, arr):
		"""
		Variant of binary search. This returns 'i' if arr[i-1] < value <= arr[i]
		"""
		low = 0
		high = len(arr)-1

		assert(value <= arr[high])

		while(low < high):
			mid = (low + high)/2
			if value <= arr[mid]:
				if mid == 0:
					return 0

				high = mid
			else:
				if arr[mid] < value <= arr[mid+1]:
					return mid+1

				low = mid

		return low