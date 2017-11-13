from sklearn import preprocessing
import pandas as pd
import gensim

""" CONFIG """
pd.options.mode.chained_assignment = None
le = preprocessing.LabelEncoder()
le.fit(["mixed", "negative", "objective", "positive"])

LabeledSentence = gensim.models.doc2vec.LabeledSentence


def load_test_file():
    test_data = pd.read_csv('../../data_test/task1-test.csv', sep='\t', names=['TweetID', 'Content'],
                            skiprows=range(1, 9))

    print('Test data set loaded with shape', test_data.shape)
    return test_data


def load_train_file():
    tweets_data = pd.read_csv('../../data_train/task1-train.csv', sep='\t', names=['TweetID', 'Content', 'Polarity'],
                              skiprows=range(1, 9))
    tweets_data['Polarity'] = transform_polarity_to_int(tweets_data['Polarity'])

    print('Train data set loaded with shape', tweets_data.shape)
    return tweets_data


def get_emoticones():
    """
    :return: A list of emoticones.
    :rtype list
    """
    with open("../../ressrc/emoticones.txt") as f:
        for i in range(8):
            next(f)
        content = f.readlines()
    # to remove whitespace characters like `\n` at the end of each line
    return [x.strip() for x in content if x.strip()]


def transform_polarity_to_int(polarity_data):
    """
    To transform class values to int.
    :param polarity_data: Values in a column of a DataFrame.
    :return:
    """
    return le.transform(polarity_data)


def inverse_transform_polarity_to_int(polarity_data):
    """
    To transform class values to int.
    :param polarity_data: Values in a column of a DataFrame.
    :return:
    """
    return le.inverse_transform(polarity_data)


def labelize_tweets(tweets, label_type):
    labelized = []
    for i, v in enumerate(tweets):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized
