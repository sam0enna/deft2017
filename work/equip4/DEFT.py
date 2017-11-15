import pandas as pd
from sitaka import sitaka
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import KeyedVectors
import numpy as np
#data
trainFileName = "../../data_train/task1-train.csv"
testFileName = "../../data_test/task1-test.csv"
Word2VecFileName = "./data/frWac.bin"

#load data
trainFile = pd.read_csv(trainFileName, sep = "\t", header = 7, names =["index","tweet","label"])
testFile = pd.read_csv(testFileName, sep = "\t", header = 7, names =["index","tweet"])


#load word2vec

#model = Word2Vec(sent_tokenize(". ".join(trainFile["tweet"])), size=100)
#model = KeyedVectors.load_word2vec_format(Word2VecFileName)
#print(model)
sitaka = sitaka()
X = []
Y = []
le = preprocessing.LabelEncoder()
le.fit(["mixed", "negative", "objective", "positive"])
Y = le.transform(trainFile["label"])
for tweet in trainFile["tweet"]:
    T = []
    tokens = sitaka.normalize(tweet)
    tag = sitaka.tag(tweet)
    T = T + sitaka.nb_syntactic_features(tag)
    T = T + sitaka.bow_features(tokens)
    T = T + sitaka.bonw_features(tokens)
    T = T + sitaka.bowo_features(tokens)
    T = T + sitaka.bowm_features(tokens)
    lemmes = sitaka.lemmes_tokens(tag)
    T.append(len(tokens) - len(lemmes))

    T.append(sitaka.polarity(sitaka.pos_polarity(lemmes), sitaka.neg_polarity(lemmes)))
    X.append(T)

X_test =[]
for tweet in testFile["tweet"]:
    T = []
    tokens = sitaka.normalize(tweet)
    tag = sitaka.tag(tweet)
    T = T + sitaka.nb_syntactic_features(tag)
    T = T + sitaka.bow_features(tokens)
    T = T + sitaka.bonw_features(tokens)
    T = T + sitaka.bowo_features(tokens)
    T = T + sitaka.bowm_features(tokens)
    lemmes = sitaka.lemmes_tokens(tag)
    T.append(len(tokens) - len(lemmes))

    T.append(sitaka.polarity(sitaka.pos_polarity(lemmes), sitaka.neg_polarity(lemmes)))
    X_test.append(T)
#classifiers = [GaussianNB(), DecisionTreeClassifier(), svm.SVC(kernel='linear', C=1), LogisticRegression()]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.33)
#for cla in classifiers:
classifier = svm.SVC(kernel='linear', C=1).fit(X,Y)
#scores = cross_val_score(classifier, X, Y, cv=10, scoring = "f1_macro")
#print("F1_Macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Classification des tweets de test...")
prediction = classifier.predict(X_test)
d = {'index': testFile["index"], 'polarité': le.inverse_transform(prediction)}
df = pd.DataFrame(data=d)

df.to_csv('./task1-run3-equip4.csv', sep='\t', header=None, index=False)