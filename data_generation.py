from sklearn.feature_extraction.text import CountVectorizer

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

def getWordsForTopic(corpus,assignments,vectorizer):
	features = vectorizer.get_feature_names()

	rows,cols = corpus.nonzero()

	for row in rows:
		print "Corpus ", row
		for col in cols:
			print features[col], ": ", assignments[row,col]

		print " ===================== "