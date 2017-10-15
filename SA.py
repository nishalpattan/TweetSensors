
# coding: utf-8

# In[80]:


import csv
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

#Function to create Training Data
def createTrainingData(corpusFile):
    trainingData={}
    with open(corpusFile,'rb') as tsvFile:
        lineReader = csv.reader(tsvFile,dialect='excel-tab')
        #print lineReader
        for row in lineReader:
            if len(row) == 2:
                sub_row = row[0].split("  ")
                trainingData[sub_row[0]] = {"label":sub_row[1],"tweet_text":row[1]}
            else:
                trainingData[row[0]] = {"label":row[1],"tweet_text":row[2]}
    return trainingData
trainingData = createTrainingData("/home/nishal/Documents/semval-task4-training data/semval-task4-training data/2016downloaded4-subtask A.tsv")
data = pd.DataFrame(trainingData)
f1_sentiment_lexicon_vader_feature = [] #feature to get positive,negative scores for tweets
senti_analyzer = SentimentIntensityAnalyzer() #creating instance for sentiment analyzer

tfidf_vector = TfidfVectorizer()#creating instance for TFIDF vectorizer
list_of_tweets = []
f2_tfidf_scores = [] #feature for tfidf scores
for only_tweet in trainingData:
    list_of_tweets.append(trainingData[only_tweet]["tweet_text"])
    
f2_tfidf_scores.append(tfidf_vector.fit_transform(list_of_tweets))

for tweet in trainingData:
    f1_sentiment_lexicon_vader_feature.append(
        {tweet:senti_analyzer.polarity_scores(trainingData[tweet]["tweet_text"])})

