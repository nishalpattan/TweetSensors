# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:54:07 2017

@author: nishal
"""

from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from twitterTokenizer import Tokenizer
import re, codecs, sys, subprocess, scipy, numpy as np, os, tempfile, math
from scipy.sparse import csr_matrix
from twitterTokenizer import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter, OrderedDict
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import string
punct = string.punctuation


def DataMatrix(ngram_features, character_gram_features,tweetText,categories):
    tokenizer_case_preserve = Tokenizer(preserve_case=True)
    tokenizer = Tokenizer(preserve_case=False)
    handmade_features, cll, cll2 = [], [], []
    for tweet in tweetText:
        feat = []
        feat.append(exclamations(tweet))
        feat.append(questions(tweet))
        feat.append(questions_and_exclamation(tweet))
        feat.append(emoticon_negative(tweet))
        feat.append(emoticon_positive(tweet))
        words = tokenizer_case_preserve.tokenize(tweet) #preserving casing
        feat.append(allCaps(words))
        feat.append(elongated(words))
        feat.append(questions_and_exclamation(words[-1]))
        handmade_features.append(np.array(feat))
        words = tokenizer.tokenize(tweet)
        words = [word.strip("_NEG") for word in words]
        cll.append(getClusters(voca_clusters, words))
        #cll2.append(getClusters(voca_handmade, words))


    bl = csr_matrix(bing_lius(tweetText, pos, different_pos_tags, pos_text))
    nrc_emo = csr_matrix(nrc_emotion(tweetText, pos, different_pos_tags, pos_text ))
    mpqa_feat = csr_matrix(mpqa(tweetText,pos, different_pos_tags, pos_text))
    handmade_features = np.array(handmade_features)
    mlb = MultiLabelBinarizer(sparse_output=True, classes = list(set(voca_clusters.values())))
    cluster_memberships_binarized = csr_matrix(mlb.fit_transform(cll))
    #mlb = MultiLabelBinarizer(sparse_output=True, classes = list(set(voca_handmade.values())))
    #cluster_memberships_binarized_2 = csr_matrix(mlb.fit_transform(cll2))
    
    hasht = csr_matrix(sent140aff(tweetText, pos, different_pos_tags, pos_text, '../lexicons/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-unigrams.txt'))
#    sent140aff_data = csr_matrix(sent140aff(tweetText, pos, different_pos_tags, pos_text, '../../lexicons/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-unigrams.txt'))
    hasht_bigrams=csr_matrix(sent140aff_bigrams(tweetText, pos, different_pos_tags, pos_text, '../lexicons/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-bigrams.txt'))
#    sent140affBigrams=csr_matrix(sent140aff_bigrams(tweetText, pos, different_pos_tags, pos_text, '../../lexicons/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-bigrams.txt'))
    sentQ = csr_matrix(get_sentiwordnet(pos_text, pos))
    pos_features = csr_matrix(pos_features)
    handmade_features = csr_matrix(handmade_features)
    # ffeatures = scipy.sparse.hstack((ngram_features, character_gram_features, cluster_memberships_binarized, handmade_features, pos_features, 
#                             sent140affBigrams, hasht_bigrams, hasht, sent140aff_data, bl, mpqa_feat, nrc_emo), dtype=float)
#    ffeatures = scipy.sparse.hstack((ngram_features, character_gram_features, cluster_memberships_binarized, handmade_features, pos_features, sent140affBigrams, hasht_bigrams, hasht, sent140aff_data, bl, mpqa_feat, nrc_emo), dtype=float)
    ffeatures = scipy.sparse.hstack((ngram_features, character_gram_features, sentQ, handmade_features, pos_features, cluster_memberships_binarized, bl, mpqa_feat, nrc_emo, hasht, hasht_bigrams ), dtype=float)

#     print ngram_features.shape, character_gram_features.shape, cluster_memberships_binarized.shape, handmade_features.shape, pos_features.shape, 
#     sent140affBigrams.shape, hasht_bigrams, hasht.shape, sent140aff_data.shape, bl.shape, mpqa_feat.shape, nrc_emo.shape
    y=[]
    for i in categories:
        if i=='positive':
            y.append(1)
        elif i == 'negative':
            y.append(-1)
        elif i == 'UNKNOWN':
            y.append(0)
        else:
            print i
    ffeatures = normalize(ffeatures)
#     ffeatures, y = shuffle(ffeatures,y)
    return ffeatures, y