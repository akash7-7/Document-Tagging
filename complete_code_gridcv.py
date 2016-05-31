import os
import re

def get_filepaths(directory):
    file_paths = [] 
    for root, directories, files in os.walk(directory):
        for filename in files:
            
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  

    return file_paths  

path = "C:\\Users\\509861\\Downloads\\reutars\\raw"
full_file_paths = get_filepaths(path)

import pandas as pd
from bs4 import BeautifulSoup,SoupStrainer

cdata = ""
for i in range(len(full_file_paths)):
    f = open(full_file_paths[i], 'r')
    data= f.read()
    cdata = cdata + data

soup = BeautifulSoup(cdata)

split = []
for link in soup.find_all('reuters'):
    split.append(link.get('lewissplit'))
    
title = [] 
for row in soup.find_all('title', class_=False):
    title.append(row.text)

date= []
for row in soup.find_all('date', class_=False):
    date.append(row.text)


topics = []
for row in soup.findAll({'topics':True,'</topics>':True}):
    line = re.sub('</d><d>',',',str(row))
    line =re.sub('</d></topics>','',line)
    line =re.sub('<d>','',line)
    line =re.sub('<topics>','',line)
    line =re.sub('</topics>','',line)
    topics.append(re.sub('<topics><d>','',line))

places = []
for row in soup.findAll({'places':True,'</places>':True}):
    line = re.sub('</d><d>',',',str(row))
    line =re.sub('</d></places>','',line)
    line =re.sub('<d>','',line)
    line =re.sub('<places>','',line)
    line =re.sub('</places>','',line)
    places.append(re.sub('<places><d>','',line))

people = []
for row in soup.findAll({'people':True,'</people>':True}):
    line = re.sub('</d><d>',',',str(row))
    line =re.sub('</d></people>','',line)
    line =re.sub('<d>','',line)
    line =re.sub('<people>','',line)
    line =re.sub('</people>','',line)
    people.append(re.sub('<people><d>','',line))


orgs = []
for row in soup.findAll({'orgs':True,'</orgs>':True}):
    line = re.sub('</d><d>',',',str(row))
    line =re.sub('</d></orgs>','',line)
    line =re.sub('<d>','',line)
    line =re.sub('<orgs>','',line)
    line =re.sub('</orgs>','',line)
    orgs.append(re.sub('<orgs><d>','',line))

exchanges = []

for row in soup.findAll({'exchanges':True,'</exchanges>':True}):
    line = re.sub('</d><d>',',',str(row))
    line =re.sub('</d></exchanges>','',line)
    line =re.sub('<d>','',line)
    line =re.sub('<exchanges>','',line)
    line =re.sub('</exchanges>','',line)
    exchanges.append(re.sub('<exchanges><d>','',line))


companies = []

for row in soup.findAll({'companies':True,'</companies>':True}):
    line = re.sub('</d><d>',',',str(row))
    line =re.sub('</d></companies>','',line)
    line =re.sub('<d>','',line)
    line =re.sub('<companies>','',line)
    line =re.sub('</companies>','',line)
    companies.append(re.sub('<companies><d>','',line))


text = []

for row in soup.find_all('text', class_=False):
    text.append(row.text)
    
df = pd.DataFrame({'date':date,'topics':topics,'places':places,'people':people,'orgs':orgs,'exchanges':exchanges,'companies':companies,
                   'text':text, 'split':split})

######################step2################################################

import os
os.chdir("C:\\Users\\509861\\Desktop")
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn import grid_search
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 


df = pd.read_csv("final_data.csv")
df = df.replace('',np.nan,regex=True)

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

from textblob import TextBlob

def split_into_lemmas(message):
    message = re.sub(r'\d+','',unicode(message, 'utf8'))
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

#df['clean_text'] = df.text.apply(split_into_lemmas)   

transformer = CountVectorizer(analyzer=split_into_lemmas, 
              lowercase=True,max_features=2000).fit(df['text'])
fitter = transformer.transform(df['text'])
tfidf_transformer = TfidfTransformer().fit(fitter)
tfidf= tfidf_transformer.transform()

one_vs_rest = OneVsRestClassifier(LinearSVC(random_state=0))
svm_places = one_vs_rest.fit(tfidf_transformer, df.places_target).predict(tfidf_transformer)

places_detector = MultinomialNB().fit(tfidf_transformer, df.places_target)
result_places = places_detector.predict(tfidf_transformer)

################yet to be done###########################################

text_train, text_test, places_train, places_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores

print scores.mean(), scores.std()

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)


%time nb_detector = grid.fit(msg_train, label_train)
print nb_detector.grid_scores_


predictions = nb_detector.predict(msg_test)
print confusion_matrix(label_test, predictions)
print classification_report(label_test, predictions)

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)


%time svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
print svm_detector.grid_scores_

print confusion_matrix(label_test, svm_detector.predict(msg_test))
print classification_report(label_test, svm_detector.predict(msg_test))
