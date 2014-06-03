from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def basicCorpus():
	texts = loadTexts()
	vectorizer = CountVectorizer(min_df = 1)
	return (vectorizer, vectorizer.fit_transform(texts))

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