from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from corpus import *

def basicCorpus():
	texts = loadTexts()
	vectorizer = CountVectorizer(min_df = 1)
	features = vectorizer.fit_transform(texts)
	return corpusFromVectorizer(features, vectorizer)

def loadTexts():
	toLoad = ['./dogs.txt', './cats.txt', 'pets.txt']
	corpus = []

	for f in toLoad:
		s = []
		with open(f, 'r') as f_handle:
			for line in f_handle:
				s.append(line)

		corpus.append(''.join(s))

	return corpus

def getWordsForTopic(corpus,assignments,vectorizer, topic):
	features = vectorizer.get_feature_names()

	rows,cols = corpus.nonzero()

	for row in set(rows):
		print "Docuent ", row
		for col in cols:
			if assignments[row,col] == topic:
				print features[col], ": ", assignments[row,col]

		print " ===================== "

def dailyKosBagOfWords():
	"""
		Translate from kos data to data my code understands.
		This is the worst thing ever. I should be punished.
	"""
	vocab = []
	with open('./dailykosvocab.txt', 'r') as vocab_handle:
		for line in vocab_handle:
			vocab.append(line)

	wordCounts = np.loadtxt('./docword.kos.txt')

	docs = []
	lastDoc = wordCounts[0,0]
	docWords = []
	for i in xrange(wordCounts.shape[0]):
		doc = wordCounts[i,0]
		wordIndex = wordCounts[i,1]
		wordCount = wordCounts[i,2]

		if doc != lastDoc:
			docs.append(' '.join(docWords))
			docWords = []
			lastDoc = doc

		for j in xrange(int(wordCount)):
			docWords.append(vocab[int(wordIndex)-1])

	return docs

def dailyKosCountMatrix(numDocs=None):
	vocab = []
	with open('./dailykosvocab.txt', 'r') as vocab_handle:
		for line in vocab_handle:
			vocab.append(line)

	wordCounts = np.loadtxt('./docword.kos.txt')

	docs = []
	lastDoc = wordCounts[0,0]
	docWords = {}
	for i in xrange(wordCounts.shape[0]):
		doc = int(wordCounts[i,0])
		wordIndex = int(wordCounts[i,1])
		wordCount = int(wordCounts[i,2])

		if doc != lastDoc:
			docs.append(docWords)
			docWords = {}
			lastDoc = doc

			if numDocs and doc > numDocs:
				break

		docWords[wordIndex-1] = wordCount

	return Corpus(docs, vocab)

def filesToCorpus(vocabFile,countFile, numDocs=None):
	vocab = []
	with open(vocabFile, 'r') as vocab_handle:
		for line in vocab_handle:
			vocab.append(line)

	wordCounts = np.loadtxt(countFile)

	docs = []
	lastDoc = wordCounts[0,0]
	docWords = {}
	for i in xrange(wordCounts.shape[0]):
		doc = int(wordCounts[i,0])
		wordIndex = int(wordCounts[i,1])
		wordCount = int(wordCounts[i,2])

		if doc != lastDoc:
			docs.append(docWords)
			docWords = {}
			lastDoc = doc

			if numDocs and doc > numDocs:
				break

		docWords[wordIndex-1] = wordCount

	return Corpus(docs, vocab)

def corpusFromVectorizer(wordCounts, vectorizer):
	vocab = vectorizer.get_feature_names()

	countMatrix = CountMatrix(numRows=wordCounts.shape[0])
	rows, cols = wordCounts.nonzero()
	for row,col in zip(rows,cols):
		countMatrix[row,col] = int(wordCounts[row,col])

	return countMatrix.toCorpus(vocab)