import os
os.chdir("C:\\Users\\509861\\Desktop")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

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

tfid = TfidfVectorizer(norm="l2",max_features=2000,stop_words = 'english',lowercase=True, ngram_range=(1, 3)) 
train_matrix = tfid.fit_transform(train.text)
train_matrix = train_matrix.todense()
test_matrix = tfid.transform(test.text)
test_matrix = test_matrix.todense()

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

pred_onevsall_places = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_matrix, train.places_target).predict(test_matrix)
acc_onevsall_places = accuracy_score(test.places_target, pred_onevsall_places)
print("category places is predicted with accuracy :")
print acc_onevsall_places

pred_onevsall_people = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_matrix, train.people_target).predict(test_matrix)
acc_onevsall_people = accuracy_score(test.people_target, pred_onevsall_people)
print("category people is predicted with accuracy :")
print acc_onevsall_people

pred_onevsall_orgs = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_matrix, train.orgs_target).predict(test_matrix)
acc_onevsall_orgs = accuracy_score(test.orgs_target, pred_onevsall_orgs)
print("category orgs is predicted with accuracy :")
print acc_onevsall_orgs

pred_onevsall_exchanges = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_matrix, train.exchanges_target).predict(test_matrix)
acc_onevsall_exchanges = accuracy_score(test.exchanges_target, pred_onevsall_exchanges)
print("category exchanges is predicted with accuracy :")
print acc_onevsall_exchanges

pred_onevsall_topics = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_matrix, train.topics_target).predict(test_matrix)
acc_onevsall_topics = accuracy_score(test.topics_target, pred_onevsall_topics)
print("category topics is predicted with accuracy :")
print acc_onevsall_topics

results= pd.DataFrame(columns=['actual_places','actual_people','actual_orgs','actual_exchanges','actual_topics','predicted_places',
                               'predicted_people','predicted_orgs','predicted_exchanges','predicted_topics'])

results['actual_places'] = test.places_target
results['actual_people'] = test.people_target
results['actual_orgs'] = test.orgs_target
results['actual_exchanges'] = test.exchanges_target
results['actual_topics'] = test.topics_target
results['predicted_places'] = pred_onevsall_places
results['predicted_people'] = pred_onevsall_people
results['predicted_orgs'] = pred_onevsall_orgs
results['predicted_exchanges'] = pred_onevsall_exchanges
results['predicted_topics'] = pred_onevsall_topics

