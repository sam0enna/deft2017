"""
author: Jonathan Bonnaud
date: November 2017
"""

import string
import numpy as np  # high dimensional vector computing library.
import os
import unicodedata
from util import *

from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

""" GLOBAL VARIABLES """
stop_words = set(stopwords.words('french'))
punctuation = set(string.punctuation) | {'’', '..', '...'}
emoticones = get_emoticones()
tokenizer = TweetTokenizer(reduce_len=True)

""" FILE PATHS """
train_file_path = '../../data_train/task1-train.csv'
test_file_path = '../../data_test/task1-test.csv'


def tokenize(tweet):
    try:
        tweet = ''.join(c.lower() for c in unicodedata.normalize('NFD', tweet)
                        if unicodedata.category(c) != 'Mn')
        tokens = tokenizer.tokenize(tweet)

        tokens = filter(lambda t: t not in stop_words, tokens)
        # tokens = filter(lambda t: not t.startswith('@'), tokens)
        # tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: t not in emoticones, tokens)
        # tokens = filter(lambda t: t not in punctuation, tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return list(tokens)
    except UnicodeError:
        return 'NC'


def postprocess(tweets_data):
    """
    Tokenize and add tokens as a feature.
    :param tweets_data:
    :return:
    """
    tweets_data['tokens'] = tweets_data['Content'].map(tokenize)
    tweets_data = tweets_data[tweets_data.tokens != 'NC']
    return tweets_data


def build_word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


""" MAIN PROGRAM """

''' Load data '''
data = load_train_file(train_file_path)
data = postprocess(data)

IS_TEST_DATA = True
if IS_TEST_DATA:
    ''' This means that the test set is available, so we load it '''
    ids_train, x_train, y_train = np.array(data.TweetID), np.array(data.tokens), np.array(data.Polarity)
    test_data = load_test_file(test_file_path)
    test_data = postprocess(test_data)
    ids_test, x_test = np.array(test_data.TweetID), np.array(test_data.tokens)
else:
    ''' Split training set to allow testing '''
    ids_train, ids_t_test, x_train, x_t_test, y_train, y_t_test = train_test_split(np.array(data.TweetID),
                                                                                   np.array(data.tokens),
                                                                                   np.array(data.Polarity),
                                                                                   test_size=0.33)

nb_examples = len(x_train)

x_train = labelize_tweets(x_train, 'TRAIN')
x_test = labelize_tweets(x_test, 'TEST') if IS_TEST_DATA else labelize_tweets(x_t_test, 'TEST')

n_dim = 200

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in x_train])
tweet_w2v.train([x.words for x in x_train], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

""" Building the sentiment classifier """

print('Building tf-idf matrix...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)  # On ne prend pas les mots qui apparaissent moins de 10 f
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('Vocab size:', len(tfidf))

train_vecs_w2v = np.concatenate([build_word_vector(z, n_dim) for z in map(lambda x: x.words, x_train)])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([build_word_vector(z, n_dim) for z in map(lambda x: x.words, x_test)])
test_vecs_w2v = scale(test_vecs_w2v)

# Classifiers tried:
classifiers = [GaussianNB(), DecisionTreeClassifier(), SVC(kernel='linear'), LogisticRegression()]
names = ["Naive Bayes", "Decision Tree", "SVM", "Logistic Regression"]

# Classifier chosen:
CLASSIFIER_ID = 3
classifier = classifiers[CLASSIFIER_ID]
name = names[CLASSIFIER_ID]

# classifier = joblib.load('./classifiers/classifier_logistic_reg.pkl')

print(name, ":")
classifier.fit(train_vecs_w2v, y_train)
# joblib.dump(classifier, './classifiers/classifier_logistic_reg.pkl')  # To save model

if not IS_TEST_DATA:
    for score in ["precision_micro", "recall_micro", "precision_macro", "recall_macro", "f1_macro"]:
        scores = cross_val_score(classifier, test_vecs_w2v, y_t_test, scoring=score, cv=10)
        print("%s: %0.2f (+/- %0.2f)" % (score,
                                         scores.mean(),
                                         scores.std() * 2))

else:
    print("Classification des tweets de test...")
    prediction = classifier.predict(test_vecs_w2v)
    d = pd.DataFrame(prediction, columns=['polarité'])
    d['polarité'] = inverse_transform_polarity_to_int(d['polarité'])
    ids = pd.DataFrame(ids_test, columns=['Id_tweet'])
    result = ids.join(d).sort_values(['Id_tweet'])

    result.to_csv('./task1-run1-equip4.csv', sep='\t', header=None, index=False)
