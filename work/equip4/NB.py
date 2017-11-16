import sys
import os
import re
import time
import string
import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
import numpy as np
import math
import nltk
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm 

classes = ["mixed", "negative", "positive","objective"]
stop_words = set(stopwords.words('french'))
punctuation = set(string.punctuation) 

def cleanText(corpus):
 
        #corpus =corpus.lower()
        corpus=replaceTwoOrMore(corpus)
        #Convert www.* or https?://* to URL
        corpus = re.sub('((www\.[^\s]+)|(http?://[^\s]+))','URL',corpus)
        corpus = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',corpus)
        #Convert @username to AT_USER
        corpus = re.sub('@[^\s]+','AT_USER',corpus)
        #Remove additional white spaces
        corpus = re.sub('[\s]+', ' ', corpus)
        #tokenization
        #tokens = tokenizer.cleanText(corpus)
        #tokens = [term.lower() for term in tokens if term.lower() not in stop_words]
        #trim
        corpus = corpus.strip('\'"')
        tokens = tokenizer.cleanText(corpus)
         # remove stopwords 
        tokens = [term.lower() for term in tokens if term.lower() not in stop_words]
        return list(tokens)

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
"""
def getFeatureVector(corpus):
    featureVector = []
    #split tweet into words
    words = corpus.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
"""
tweets_data = pd.read_csv('data/task1-train.csv', sep='\t', names=['TweetID', 'Content', 'Polarity'],
                              skiprows=range(1, 9))
tests_data = pd.read_csv('data/task1-test.csv', sep='\t', names=['TweetID', 'Content'], skiprows=range(1, 9))


def postprocess(tweets_data):
 
    tweets_data['tokens'] = tweets_data['Content']
    tweets_data = tweets_data[tweets_data.tokens != 'NC']
    return tweets_data


y_train=tweets_data['Polarity']
y_test =["mixed", "negative", "positive","objective"] 

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(tweets_data['Content'])
test_matrix = vectorizer.transform(tests_data['Content'])
#print (train_matrix .todense())

model1 = MultinomialNB()
#t0 = time.time()
model1.fit(train_matrix,y_train)
#t1 = time.time()
result1 = model1.predict(test_matrix)
#t2 = time.time()
#time_rbf_train = t1-t0
#time_rbf_predict = t2-t1
#print(result1)
model2 = LinearSVC()
model2.fit(train_matrix,y_train)
result2 = model2.predict(test_matrix)

""""for score in ["precision_micro", "recall_micro", "precision_macro", "recall_macro", "f1_macro"]:
        scores = cross_val_score(model1, test_matrix, y_test, scoring=score, cv=10)
        print("%s: %0.2f (+/- %0.2f)" % (score,
                                         scores.mean(),
                                         scores.std() * 2)) """

run1 = pd.DataFrame(result1, index=tests_data['TweetID'])
run2 = pd.DataFrame(result2, index=tests_data['TweetID'])

run1.to_csv("task1-run1-equip4.csv", sep='\t', header=False)
run2.to_csv("task1-run2-equip4.csv", sep='\t', header=False)
