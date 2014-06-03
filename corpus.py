class CountMatrix:

	def __init__(self, data):
		self.__data = data

	def __getitem__(self, pos):
		row,col = pos
		return self.__data[row].get(col, 0)

	def __setitem__(self, pos, value):
		row,col = pos
		self.__data[row][col] = value

	def elements(self):
		for i in xrange(len(self.__data)):
			for key in self.__data[i]:
				yield (key, self.__data[i][key])

	def nonzero(self):
		for i in xrange(len(self.__data)):
			for key in self.__data[i]:
				yield (i, key)