# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:09:55 2017

@author: nishal
"""
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
import pandas as pd
import nltk
import re
import codecs
from twitterTokenizer import Tokenizer
from collections import Counter
import sys, subprocess, scipy, numpy as np, os, tempfile, math
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import string
punct = string.punctuation

def PreProcessingTweets(tweet_text): #input tweet_text is list of tweets
    processed_lists=[]
    for i in tweet_text:
        try:
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',i) #substituting urls with constant URL
            tweet = re.sub('@[^\s]+','AT_USER',tweet) #substituting user mentions with constant AT_USER
            tweet = re.sub(r'#([^\s]+)',r'\1',tweet)
            processed_lists.append(tweet)
        except TypeError:
            print "Error Occured"
            tweet = re.sub(r'#([^\s]+)',r'\1',i)
            processed_lists.append(tweet)
        except:
            print "Error Occured"
            tweet = re.sub('@[^\s]+','AT_USER',i) #substituting user mentions with constant AT_USER
            tweet = re.sub(r'#([^\s]+)',r'\1',tweet)
            processed_lists.append(tweet)
    return processed_lists
def FeatureExtraction(tweet_text_list):
    index=0
    for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
        exclamtionCount = feature_excalamation_count(tweet)
        questionCount = feature_question_count(tweet)
        Exclamation_Question_Count=feature_excalamation_and_question_count(tweet)
        ElongatedCount=feature_elongated_count(tweet)
        Positive_emoticonCount = feature_positive_emoticon_count(tweet)
        Negative_emoticonCount = feature_negative_emoticon_count(tweet)
        emoticon_existsBinary = feature_emoticon_exists_binary(tweet)
        features.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
                            Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary)
        index+=1
    ngrams_train = feature_ngrams(tweet_text_list)
    charGrams_train = feature_char_grams(tweet_text_list)
    #features['n-grams'] = ngrams
    #features['char-grams']=charGrams
    return features,ngrams_train,charGrams_train
def FeatureExtraction_test(tweet_text_list):
    index=0
    for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
        exclamtionCount = feature_excalamation_count(tweet)
        questionCount = feature_question_count(tweet)
        Exclamation_Question_Count=feature_excalamation_and_question_count(tweet)
        ElongatedCount=feature_elongated_count(tweet)
        Positive_emoticonCount = feature_positive_emoticon_count(tweet)
        Negative_emoticonCount = feature_negative_emoticon_count(tweet)
        emoticon_existsBinary = feature_emoticon_exists_binary(tweet)
        features_test.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
                            Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary)
        index+=1
    #ngrams_train = feature_ngrams(tweet_text_list)
    #charGrams_train = feature_char_grams(tweet_text_list)
    #features['n-grams'] = ngrams
    #features['char-grams']=charGrams
    return features_test
def FeatureExtraction_final_test(tweet_text_list):
    index=0
    for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
        exclamtionCount = feature_excalamation_count(tweet)
        questionCount = feature_question_count(tweet)
        Exclamation_Question_Count=feature_excalamation_and_question_count(tweet)
        ElongatedCount=feature_elongated_count(tweet)
        Positive_emoticonCount = feature_positive_emoticon_count(tweet)
        Negative_emoticonCount = feature_negative_emoticon_count(tweet)
        emoticon_existsBinary = feature_emoticon_exists_binary(tweet)
        final_features_test.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
                            Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary)
        index+=1
    #ngrams_train = feature_ngrams(tweet_text_list)
    #charGrams_train = feature_char_grams(tweet_text_list)
    #features['n-grams'] = ngrams
    #features['char-grams']=charGrams
    return final_features_test
    
def feature_ngrams(tweet_text):   #method to get ngrams (n = [1,2,3,4,..]) of a tweet in taining set
    ngram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(1,4), stop_words=None, lowercase=True,  tokenizer=tokenizer.tokenize, n_features=10000)
    ngram_features = ngram.fit_transform(tweet_text)
    return ngram_features
    #return list(nltk.ngrams(tweet_text,n))
def feature_ngrams_test(tweet_text):
    ngram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(1,4), stop_words=None, lowercase=True,  tokenizer=tokenizer.tokenize, n_features=10000)
    ngram_features = ngram.fit_transform(tweet_text)
    return ngram_features
def feature_char_grams(tweet_text):   #method to get ngrams (n = [1,2,3,4,..]) of a tweet in a training set
    char_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000)
    char_gram_features = char_gram.fit_transform(tweet_text)
    return char_gram_features
def feature_charGrams_test(tweet_text):
    char_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000)
    char_gram_features = char_gram.fit_transform(tweet_text)
    return char_gram_features
def feature_excalamation_count(tweet_text): #method to get frequency of exclamation(!) in a tweet
    return tweet_text.count("!")
    
def feature_question_count(tweet_text):  #method to get frequency of question(?) in a tweet
    return tweet_text.count("?")
    
def feature_excalamation_and_question_count(tweet_text):#method to get frequency of question mark(?) & exclamation (!) in a tweet
    return tweet_text.count("?") + tweet_text.count("!")
    
def feature_elongated_count(tweet_text): #method to get frequency of elongated words and capitalized words
    count_capitalized=0
    count_elongated=0
    for i in tweet_text.split():
        if i.isupper():
            count_capitalized+=1
        pattern = re.compile(r'(.)\1?')
        result = [x.group() for x in pattern.finditer(i)]
        filtered_result = [x for x in result if len(x) == 2]
        if len(filtered_result) > 2:
            count_elongated+=1
        return count_capitalized+count_elongated
def feature_positive_emoticon_count(tweet_text): #method to get frequency of positive emoticons in a tweet
    emoticons_pos = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # nose
      [\)\]dD\}@]                # mouth      
      |                          # reverse order now! 
      [\)\]dD\}@]                # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
    capture_pos_emoticon=re.compile(emoticons_pos,re.VERBOSE|re.I|re.UNICODE)
    capture_pos_emoticon_list=capture_pos_emoticon.findall(tweet_text)
    return len(capture_pos_emoticon_list)
    
def feature_negative_emoticon_count(tweet_text): #method to get frequency of negative emoticons in a tweet
    emoticons_neg = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\(\[pP/\:\{\|] # mouth      
      |                          # reverse order now! 
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      )"""
    capture_neg_emoticon=re.compile(emoticons_neg,re.VERBOSE|re.I|re.UNICODE)
    capture_neg_emoticon_list=capture_neg_emoticon.findall(tweet_text)
    return len(capture_neg_emoticon_list)
 #method which checks the existence of emoticon in a tweet and returns a binary
def feature_emoticon_exists_binary(tweet_text):
    if feature_negative_emoticon_count(tweet_text) or feature_positive_emoticon_count(tweet_text):
        return 1
    else:
        return 0

def get_pos_tags_and_hashtags(tweetText):
    tf = tempfile.NamedTemporaryFile(delete=False)
    with codecs.open(tf.name, 'w', encoding='utf8') as out:
        for i in tweetText:
            #out.write("%s\n"%i.decode('utf-8'))
            out.write("%s\n"%i)
    com = "/home/nishal/ALL_FILES_HERE/CS_COURSES/NLP/SemEval_Task4/ark-tweet-nlp-0.3.2/runTagger.sh %s"%tf.name
    op= subprocess.check_output(com.split())
    op = op.splitlines()
    pos_text = [x.split("\t")[0].split() for x in op]
    pos = [x.split("\t")[1].split() for x in op]
    different_pos_tags = list(set([x for i in pos for x in i]))
    pos_features = []
    for instance in pos:
        tags = []
        instance = Counter(instance)
        for pos_tag in different_pos_tags:
            try:
                tags.append(instance[pos_tag])
            except:
                tags.append(0)
        pos_features.append(np.array(tags))
    pos_features = np.array(pos_features)
    #print "------------\nPOS-tagging finished!\n------------\nThere are %d pos-tags (incl. hashtags). Shape: %d,%d"%(len(different_pos_tags), pos_features.shape[0],  pos_features.shape[1])
    for key1, i in enumerate(pos_text):
        flag = False
        for key, j in enumerate(i):
            i[key] = j.lower()
            if flag:
                if pos[key1][key] in "AVRN" :
                    i[key]+="_NEG"
                else:
                    flag=False
            if j in negation:
                flag = True
    os.remove(tf.name)
    return pos, pos_features, different_pos_tags, pos_text


def DataMatrix(ngram_features,
               character_gram_features,
               general_features,
               pos,
               pos_features,
               different_pos_tags,
               pos_text,
               categories):
    #print"LENGTH", len(ngram_features),len(character_gram_features),len(pos_features)
    pos_features = csr_matrix(pos_features)
    Final_Features = scipy.sparse.hstack((ngram_features, 
                                          character_gram_features,
                                          general_features.as_matrix(columns=["exclamation_count","question_count",
                                                                              "exclamation_question_count","capital_and_elongated_count",
                                                                              "positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"]), 
                                          pos_features),dtype=float)
    y=[]
    for i in range(len(categories)):
        if categories[i]=='positive':
            y.append(1)
        elif categories[i] == 'negative':
            y.append(-1)
        elif categories[i] == 'UNKNOWN':
            y.append(0)
        else:
            print categories[i]
    Final_Features= normalize(Final_Features)
    return Final_Features, y
def KLD(true, pred):
    epsilon = 0.5 / len(pred)
    countsTrue, countsPred = Counter(true), Counter(pred)
    p_pos = countsTrue[0]/len(true)
    p_neg = countsTrue[1]/len(true)
    est_pos = countsPred[0]/len(true)
    est_neg = countsPred[1]/len(true)
    p_pos_s = (p_pos + epsilon)/(p_pos+p_neg+2*epsilon)
    p_neg_s = (p_neg + epsilon)/(p_pos+p_neg+2*epsilon)
    est_pos_s = (est_pos+epsilon)/(est_pos+est_neg+2*epsilon)
    est_neg_s = (est_neg+epsilon)/(est_pos+est_neg+2*epsilon)
    return p_pos_s*math.log10(p_pos_s/est_pos_s)+p_neg_s*math.log10(p_neg_s/est_neg_s)
def showMyKLD(true, pred, l):
    s= []
    for key, val in enumerate(l):
        if key == len(l)-1:
            break
        s.append(KLD(true[val:l[key+1]], pred[val:l[key+1]]))
    return sum(s)/len(s)




negation = set(["never","no","nothing","nowhere","noone","none","not","havent","haven't","hasnt","hasn't","hadnt","hadn't", 
                "cant","can't","couldnt","couldn't","shouldnt","shouldn't","wont","won't","wouldnt","wouldn't","dont","don't","doesnt","doesn't","didnt",
                "didn't","isnt","isn't","arent","aren't","aint","ain't"])
tokenizer = Tokenizer()
#Load Train data
data=pd.read_csv("/home/nishal/ALL_FILES_HERE/CS_COURSES/NLP/SemEval_Task4/semval-task4-training data/semval-task4-training data/2016-part2-subtaskBD.tsv",
                 sep='\t',
                 names=["tweet_id","topic","label","tweet_text"])
#Load Dev Data
dev_data=data=pd.read_csv("/home/nishal/ALL_FILES_HERE/CS_COURSES/NLP/SemEval_Task4/semval-task4-training data/semval-task4-training data/subtaskBD.dev.downloaded.all.tsv",
                 sep='\t',
                 names=["tweet_id","topic","label","tweet_text"])
#Load DEV-TEST data
gold_test_data = codecs.open("/home/nishal/ALL_FILES_HERE/CS_COURSES/NLP/SemEval_Task4/semval-task4-training data/semval-task4-training data/100_topics_XXX_tweets.topic-two-point.subtask-BD.devtest.gold.downloaded.txt").read().splitlines()
gold_test_data=[i.split("\t") for i in gold_test_data if i.split("\t")[-1] != 'Not Available']
gold_test_data = pd.DataFrame(gold_test_data,columns=["tweet_id","topic","label","tweet_text"])

#Final Test Data
final_test_data = codecs.open("/home/nishal/ALL_FILES_HERE/CS_COURSES/NLP/SemEval_Task4/semval-task4-training data/semval-task4-training data/twitter-2016test-BD.txt").read().splitlines()
final_test_data=[i.split("\t") for i in final_test_data if i.split("\t")[-1] != 'Not Available']
for i in final_test_data:
    if '' in i:
        i.remove('')
final_test_data = pd.DataFrame(final_test_data,columns=["tweet_id","topic","label","tweet_text"])
final_test_data = final_test_data[final_test_data.tweet_text != None]

#Cleaning & PreProcessing Data
#remove rows with tweet_text = "Not Available"
data = data[data.tweet_text != "Not Available"]
dev_data = dev_data[dev_data.tweet_text != "Not Available"]
data = data.append(dev_data,ignore_index=True)
data['tweet_text']=PreProcessingTweets(data['tweet_text']) #Cleaning Tweets--> remove urls,user mentions
tweet_train,categories_train = list(data['tweet_text']),list(data['label'])
gold_test_data['tweet_text']=PreProcessingTweets(gold_test_data['tweet_text'])
tweet_test,categories_test = list(gold_test_data['tweet_text']),list(gold_test_data['label'])
final_test_data['tweet_text']=PreProcessingTweets(final_test_data['tweet_text'])
tweet_final_test=list(final_test_data['tweet_text'])

features = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"])

features_test = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"])

final_features_test = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"])

topic_label = [gold_test_data['topic'][i] for i in range(len(gold_test_data['topic']))] #This is to group tweets by topic. Can by improved!!
cnt = Counter(topic_label) #gives the frequency of each label
yo = [0]
test_cats = []
for i in range(len(set(topic_label))):
    num = cnt[topic_label[yo[i]]]
    test_cats.append(topic_label[num+yo[i]-1])
    yo.append(num+yo[i])
#GET FEATURES FOR TRAINING
generalFeatures,nGram_features_train,charGram_features_train=FeatureExtraction(data['tweet_text'])
pos1, pos_features1, different_pos_tags1, pos_text1 = get_pos_tags_and_hashtags(tweet_train+tweet_test) #Get POS of everything
pos, pos_features, different_pos_tags, pos_text =  pos1[:len(categories_train)], pos_features1[:len(categories_train)], different_pos_tags1, pos_text1[:len(categories_train)] #Split train-test again
pos_test, pos_features_test, different_pos_tags_test, pos_text_test = pos1[:len(categories_test)], pos_features1[:len(categories_test)], different_pos_tags1, pos_text1[:len(categories_test)] #Split train-test again
nGram_features_train.data **= 0.7 #a-power transformation
charGram_features_train.data **= 0.7 #a-power transformation

generalFeatures_test = FeatureExtraction_test(gold_test_data['tweet_text'])
nGram_features_test = feature_ngrams_test(gold_test_data['tweet_text'])
charGram_features_test = feature_charGrams_test(gold_test_data['tweet_text'])
nGram_features_test.data **= 0.7
charGram_features_test.data **= 0.7


generalFeatures_final_test = FeatureExtraction_final_test(final_test_data['tweet_text'])
nGram_features_final_test = feature_ngrams_test(final_test_data['tweet_text'])
charGram_features_final_test = feature_charGrams_test(final_test_data['tweet_text'])
pos_final_test, pos_features_final_test, different_pos_tags_final, pos_final_test= get_pos_tags_and_hashtags(tweet_final_test) #Get POS of everything


nGram_features_final_test.data **= 0.7
charGram_features_final_test.data **= 0.7

x_train, y_train = DataMatrix(nGram_features_train, 
                              charGram_features_train, 
                              generalFeatures,
                              pos,
                              pos_features,
                              different_pos_tags,
                              pos_text,
                              data['label']) #Combine all  features (train)
x_test,y_test = DataMatrix(nGram_features_test,
                           charGram_features_test,
                           generalFeatures_test,
                           pos_test,
                           pos_features_test, 
                           different_pos_tags_test, 
                           pos_text_test,
                           gold_test_data['label'])
              
x_final_test,y_final_test = DataMatrix(nGram_features_final_test,
                           charGram_features_final_test,
                           generalFeatures_final_test,
                           pos_final_test,
                           pos_features_final_test,
                           different_pos_tags_final,
                           pos_final_test,
                           final_test_data['label'])
print "FINAL TEST SUBMISSION"
for c in np.logspace(0,100): #used 100 for submission
    clf = svm.LinearSVC(C=c, loss='squared_hinge', penalty='l2', class_weight='balanced', multi_class='crammer_singer', max_iter=4000, dual=True, tol=1e-6)
    clf.fit(x_train, y_train)
    print "KLD---> c---->",  showMyKLD(y_test, clf.predict(x_test),yo), c
    print " FINAL TEST  --> KLD--> c-->",showMyKLD(y_final_test, clf.predict(x_final_test),yo), c
    """
for c in np.logspace(0,1): 
    clf = svm.LinearSVC(C=c, loss='squared_hinge', penalty='l2', class_weight='balanced', multi_class='crammer_singer', max_iter=4000, dual=True, tol=1e-6)
    clf.fit(x_train, y_train)
    print "KLD---> c---->",  showMyKLD(y_test, clf.predict(x_test),yo), c
    
print "FIANL TEST"
print "KLD---> c---->",  showMyKLD(y_test, clf.predict(x_test),yo), c
"""


