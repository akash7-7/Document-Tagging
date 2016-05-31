#Final model for Places

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

train = df.query('split=="TRAIN"')
test = df.query('split=="TEST"')
unused = df.query('split=="NOT-USED"')

def split_into_lemmas(article):
    article = re.sub(r'\d+','',unicode(article, 'utf8'))
    words = TextBlob(article).words.lower()
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

tfid = TfidfVectorizer(norm="l2",analyzer=split_into_lemmas,max_features=100,
	                   stop_words = 'english',ngram_range=(1, 3)) 
#tfid1 = TfidfVectorizer(norm="l2",max_features=100,stop_words = 'english',lowercase=True, ngram_range=(1, 3),max_df=1.0,min_df=0.5)
train_matrix = tfid.fit_transform(train.text)
train_matrix = train_matrix.todense()
test_matrix = tfid.transform(test.text)
test_matrix = test_matrix.todense()

#Modelling

X = train_matrix
y = train.places_target

#svm places
one_vs_rest = OneVsRestClassifier(LinearSVC(random_state=0))
scores = cross_validation.cross_val_score(one_vs_rest, X, y, cv=5, scoring='accuracy')
print("svm cross validation scores mean for places is %0.2f" % scores.mean())
svm_places = one_vs_rest.fit(train_matrix, train.places_target).predict(test_matrix)

#OutPutCodeClassifier
clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("OutputCodeClassifier cross validation scores mean for places is %0.2f" % scores.mean())
output_places = clf.fit(train_matrix, train.places_target).predict(test_matrix)

#KNN
knn = KNeighborsClassifier()
scores = cross_validation.cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print("knn cross validation scores mean for places is %0.2f" % scores.mean())
knn_places = knn.fit(train_matrix, train.places_target).predict(test_matrix)

#Naive_bayes
NB = MultinomialNB()
scores = cross_validation.cross_val_score(NB, X, y, cv=5, scoring='accuracy')
print("naivebayes cross validation scores mean for places is %0.2f" % scores.mean())
nb_places = NB.fit(train_matrix, train.places_target).predict(test_matrix)

#Random Forests
rfc = RandomForestClassifier(n_estimators=10)
scores = cross_validation.cross_val_score(rfc, X, y, cv=5, scoring='accuracy')
print("Random Forest cross validation scores mean for places is %0.2f" % scores.mean())
rf_places = rfc.fit(train_matrix, train.places_target).predict(test_matrix)

places = test.places_target

predictions = pd.DataFrame({"SVM":svm_places,"KNN":knn_places,"NB":nb_places,"RF":rf_places,"Actual":places})
predictions.to_csv("places_predictions.csv")

import difflib
out_match = 0
svm_match = 0
knn_match = 0
nb_match = 0
rf_match = 0

for i in range(len(places)):
	svm_match =  svm_match + difflib.SequenceMatcher(None,svm_places[i],places[i]).ratio()
	out_match = out_match + difflib.SequenceMatcher(None,svm_places[i],places[i]).ratio()
	knn_match =  knn_match + difflib.SequenceMatcher(None,knn_places[i],places[i]).ratio()
	nb_match = nb_match + difflib.SequenceMatcher(None,nb_places[i],places[i]).ratio()
	rf_match = rf_match + difflib.SequenceMatcher(None,rf_places[i],places[i]).ratio()

print(float(svm_match)/len(places))
print(float(out_match)/len(places))
print(float(knn_match)/len(places))
print(float(nb_match)/len(places))
print(float(rf_match)/len(places))



from sklearn.linear_model import SGDClassifier
text_clf = SGDClassifier(loss='hinge', penalty='l2',
	alpha=1e-3, n_iter=5, random_state=42)
sgd_places = text_clf.fit(train_matrix,train.places_target).predict(test_matrix)

import difflib
sgd_match = 0
for i in range(len(places)):
	sgd_match = sgd_match + difflib.SequenceMatcher(None,sgd_places[i],places[i]).ratio()

print(float(sgd_match)/len(places))
