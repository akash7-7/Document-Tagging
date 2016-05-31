import os
os.chdir("C:\\Users\\509861\\Desktop")
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

df = pd.read_csv("final_data.csv")
df = df.replace('',np.nan,regex=True)

df.ix[df.places !=' ','places_flag'] = 0
df.ix[df.people !=' ','people_flag'] = 0
df.ix[df.topics !=' ','topics_flag'] = 0
df.ix[df.orgs !=' ','orgs_flag'] = 0
df.ix[df.exchanges !=' ','exchanges_flag'] = 0
df.loc[df['places'].notnull(), 'places_flag'] = 1
df.loc[df['people'].notnull(), 'people_flag'] = 2
df.loc[df['topics'].notnull(), 'topics_flag'] = 3
df.loc[df['orgs'].notnull(), 'orgs_flag'] = 4
df.loc[df['exchanges'].notnull(), 'exchanges_flag'] = 5

places_target= []
for x in range(len(df.places)):
    places_target.append(df.places[x])
#print(places_target)
df['places_target'] = places_target

people_target= []
for x in range(len(df.people)):
    people_target.append(df.people[x])
#print(people_target)
df['people_target'] = people_target

topics_target= []
for x in range(len(df.topics)):
    topics_target.append(df.topics[x])
#print(topics_target)
df['topics_target'] = topics_target

orgs_target= []
for x in range(len(df.orgs)):
    orgs_target.append(df.orgs[x])
#print(orgs_target)
df['orgs_target'] = orgs_target

exchanges_target= []
for x in range(len(df.exchanges)):
    exchanges_target.append(df.exchanges[x])
#print(exchanges_target)
df['exchanges_target'] = exchanges_target

df['places_target'] = df['places_target'].astype(str)
df['people_target'] = df['people_target'].astype(str)
df['topics_target'] = df['topics_target'].astype(str)
df['orgs_target'] = df['orgs_target'].astype(str)
df['exchanges_target'] = df['exchanges_target'].astype(str)

df.loc[(df['places_target'] == '0')] ='blank'
df.loc[(df['people_target'] == '0')] ='blank'
df.loc[(df['topics_target'] == '0')] ='blank'
df.loc[(df['orgs_target'] == '0')] ='blank'
df.loc[(df['exchanges_target'] == '0')] ='blank'

def split_into_lemmas(article):
    article = re.sub(r'\d+','',unicode(article, 'utf8'))
    words = TextBlob(article).words.lower()
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

def split_into_tokens(article):
    article = unicode(article, 'utf8')  # convert bytes into proper unicode
    return TextBlob(article).words

%matplotlib inline
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

train = df.query('split=="TRAIN"')
test = df.query('split=="TEST"')
unused = df.query('split=="NOT-USED"')

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(norm="l2",analyzer=split_into_lemmas,max_features=200,
                       stop_words = 'english',ngram_range=(1, 3)) ),
    ('classifier', OneVsRestClassifier(LinearSVC(random_state=0))), 
])


scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         train['text'],  # training data
                         train['places_target'],  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores

print scores.mean(), scores.std()

params = {
    'tfidf__use_idf': (True, False),
    'analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)
