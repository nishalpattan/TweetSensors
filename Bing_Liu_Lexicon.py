class Bing_Liu_Lexicon():
	def __init__(self):
		self.fp_positive = '/home/nishal/ALL_FILES_HERE/CS_COURSES/NLP/SemEval_Task4/positive-words.txt'
		self.fp_negative = '/home/nishal/ALL_FILES_HERE/CS_COURSES/NLP/SemEval_Task4/negative-words.txt'
		self.d = {}
	def construct_lexicons(self):
		fp_positive=open(self.fp_positive)
		fp_negative=open(self.fp_negative)
		positive_words=fp_positive.read().split('\n')
		negative_words=fp_negative.read().split('\n')
		self.d['positive_words'] = positive_words
		self.d['negative_words'] = negative_words
		return self.d
		
	def get_features(self,tokenized_tweet):
		positive_count = 0
		negative_count = 0
		for word in tokenized_tweet:
			try:
				if word in self.d['positive_words']:
					positive_count+=1
				elif word in self.d['negative_words']:
					negative_count+=1
			except Exception as e:
				#print e
				pass
		return [positive_count,negative_count]
if __name__ == "__main__":
	bing_liu = Bing_Liu_Lexicon()
	d = bing_liu.construct_lexicons()
	print len(d['positive_words'])
	print len(d['negative_words'])


