from corpus import *

def nipsCorpus(numDocs=None):
	nipsVocab = 'data/nipsvocab.txt'
	nipsCounts = 'data/docword.nips.txt'
	return filesToCorpus(nipsVocab,nipsCounts,numDocs)

def dailyKosCorpus(numDocs=None):
	kosVocab = 'data/dailykosvocab.txt'
	kosCounts = 'data/docword.kos.txt'
	return filesToCorpus(kosVocab,kosCounts,numDocs)

def filesToCorpus(vocabFile,countFile, numDocs=None):
	
	vocab = []
	with open(vocabFile, 'r') as vocab_handle:
		for line in vocab_handle:
			vocab.append(line)
	
	with open(countFile, 'r') as count_handle:
		#First three numbers are D, V, and N
		[docCount,wordCount,totalCounts] = [int(count_handle.readline()) \
			for _ in xrange(3)]

		if numDocs and docCount > numDocs:
			docCount = numDocs

		countMatrix = CountMatrix(numRows=docCount)
		for line in count_handle:
			[doc,word,count] = [int(x) for x in line.split()]

			#docs are not zero indexed in files
			doc -= 1
			#neither are words
			word -= 1

			if numDocs and doc >= numDocs:
				break

			countMatrix[doc,word] = count

	return countMatrix.toCorpus(vocab)