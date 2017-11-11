"""
author: Jonathan Bonnaud
"""
import gensim
import string
import numpy as np  # high dimensional vector computing library.
import os
import pandas as pd  # provide sql-like data manipulation tools. very handy.
import unicodedata
import equip4.util as util

from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

""" CONFIG """
pd.options.mode.chained_assignment = None
LabeledSentence = gensim.models.doc2vec.LabeledSentence
tokenizer = TweetTokenizer(reduce_len=True)

""" GLOBAL VARIABLES """
stop_words = set(stopwords.words('french'))
punctuation = set(string.punctuation) | {'’', '..', '...'}
emoticones = util.get_emoticones()

le = preprocessing.LabelEncoder()
le.fit(["mixed", "negative", "objective", "positive"])


def load_test_file():
    test_data = pd.read_csv('../data_test/task1-test.csv', sep='\t', names=['TweetID', 'Content'],
                            skiprows=range(1, 9))

    print('Test data set loaded with shape', test_data.shape)
    return test_data


def transform_polarity_to_int(polarity_data):
    """
    To transform class values to int.
    :return:
    """
    return le.transform(polarity_data)


def inverse_transform_polarity_to_int(polarity_data):
    """
    To transform class values to int.
    :return:
    """
    return le.inverse_transform(polarity_data)


def load_train_file():
    tweets_data = pd.read_csv('../data_train/task1-train.csv', sep='\t', names=['TweetID', 'Content', 'Polarity'],
                              skiprows=range(1, 9))
    tweets_data['Polarity'] = transform_polarity_to_int(tweets_data['Polarity'])

    print('data set loaded with shape', tweets_data.shape)
    return tweets_data


def tokenize(tweet):
    try:
        tweet = ''.join(c.lower() for c in unicodedata.normalize('NFD', tweet)
                        if unicodedata.category(c) != 'Mn')
        tokens = tokenizer.tokenize(tweet)
        # print(tokens)

        tokens = filter(lambda t: t not in stop_words, tokens)
        # tokens = filter(lambda t: not t.startswith('@'), tokens)
        # tokens = filter(lambda t: not t.startswith('#'), tokens)
        # tokens = filter(lambda t: t not in emoticones, tokens)
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
    tqdm.pandas(desc="Tokenizing")  # Add a label to the progress map
    tweets_data['tokens'] = tweets_data['Content'].progress_map(tokenize)
    tweets_data = tweets_data[tweets_data.tokens != 'NC']
    return tweets_data


def labelize_tweets(tweets, label_type):
    labelized = []
    for i, v in enumerate(tweets):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


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


data = load_train_file()
# print(data[:5])
data = postprocess(data)
# print(data[:5])

y_test = None
if os.path.exists('../data_test/task1-test.csv'):
    x_train, y_train = np.array(data.tokens), np.array(data.Polarity)
    test_data = load_test_file()
    test_data = postprocess(test_data)
    ids_test, x_test = np.array(test_data.TweetID), np.array(test_data.tokens)
else:
    ids_train, ids_test, x_train, x_test, y_train, y_test = train_test_split(np.array(data.TweetID),
                                                                             np.array(data.tokens),
                                                                             np.array(data.Polarity), test_size=0.33)

nb_examples = len(x_train)

x_train = labelize_tweets(x_train, 'TRAIN')
x_test = labelize_tweets(x_test, 'TEST')

# print(x_train[0])

n_dim = 200

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in x_train])
tweet_w2v.train([x.words for x in x_train], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

# print(tweet_w2v['bien'])
# print(tweet_w2v.most_similar('content'))

""" Building a sentiment classifier """

print('Building tf-idf matrix...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('Vocab size:', len(tfidf))

train_vecs_w2v = np.concatenate([build_word_vector(z, n_dim) for z in map(lambda x: x.words, x_train)])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([build_word_vector(z, n_dim) for z in map(lambda x: x.words, x_test)])
test_vecs_w2v = scale(test_vecs_w2v)

# classifier = joblib.load('./classifiers/classifier_logistic_reg.pkl')

classifier = LogisticRegression()
# classifier = svm.SVC(kernel='linear')
classifier.fit(train_vecs_w2v, y_train)
joblib.dump(classifier, './classifiers/classifier_logistic_reg.pkl')

if y_test is not None:
    # score = classifier.score(test_vecs_w2v, y_test) OR
    scores = cross_val_score(classifier, train_vecs_w2v, y_train, cv=10, scoring='precision_micro')
    # ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
    # 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision',
    # 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
    # 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

prediction = classifier.predict(test_vecs_w2v)
d = pd.DataFrame(prediction, columns=['polarité'])
d['polarité'] = inverse_transform_polarity_to_int(d['polarité'])
ids = pd.DataFrame(ids_test, columns=['Id_tweet'])
result = ids.join(d)
# result.set_index('Id_tweet', inplace=True)

result.to_csv('./task1-run1-equip4.csv', sep='\t', header=None, index=False)
