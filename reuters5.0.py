import os
import re

def get_filepaths(directory):
    file_paths = [] 
    for root, directories, files in os.walk(directory):
        for filename in files:
            
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  

    return file_paths  

path = "C:\\Users\\Akash Bhople\\Desktop\\reuters\\raw"
full_file_paths = get_filepaths(path)
                
import pandas as pd
from bs4 import BeautifulSoup,SoupStrainer

cdata = ""
for i in range(len(full_file_paths)):
    f = open(full_file_paths[i], 'r')
    data= f.read()
    cdata= cdata + data
                
soup = BeautifulSoup(cdata)
#print soup.prettify()[0:5000]
                
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


text = []

for row in soup.find_all('text', class_=False):
    text.append(row.text)
    
split = []
for link in soup.find_all('reuters'):
    split.append(link.get('lewissplit'))

df = pd.DataFrame({'date':date,'topics':topics,'places':places,'people':people,'orgs':orgs,'exchanges':exchanges,'text':text, 'split':split})

#df = pd.read_csv('reuters_data.csv')

#df['target'] = df['places'] 

#df['target'] = df.target.map(str) + ',' + df.people.map(str) + ',' + df.companies.map(str) + ','  +  df.exchanges.map(str) + ',' + df.orgs.map(str) 

#df['target'] = df['target'].str.replace(',nan','')

#df['target'] = df['target'].str.replace('nan','')

#df['target'] = df['target'].str.replace(',',' ')

import numpy as np
df1 = df
df1 = df1.replace('',np.nan,regex=True)
#df1.head()

df2 = df1

#df1.describe()

#df1.text.isnull().any().any()

#df1.orgs.isnull().sum()

train = df1.query('split=="TRAIN"')
test = df1.query('split=="TEST"')
unused = df1.query('split=="NOT-USED"')  

df1.ix[df1.places !=' ','places_flag'] = 2
#df1.head()
df1.places_flag.value_counts()
#print df1.places.isnull().sum()
#print df1.people.isnull().sum()
#print df1.topics.isnull().sum()
#print df1.exchanges.isnull().sum()
#print df1.orgs.isnull().sum()

df2.loc[df2['places'].notnull(), 'places_flag'] = 1
#print(df2['places'].notnull())
#df2.head()
df2.places_flag.value_counts()

df2.loc[df2['people'].notnull(), 'people_flag'] = 2
#print(df2['people'].notnull())
#df2.head()
df2.people_flag.value_counts()

df2.loc[df2['topics'].notnull(), 'topics_flag'] = 3
#print(df2['topics'].notnull())
#df2.head()
df2.topics_flag.value_counts()

df2.loc[df2['orgs'].notnull(), 'orgs_flag'] = 4
#print(df2['orgs'].notnull())
#df2.head()
df2.orgs_flag.value_counts()

df2.loc[df2['exchanges'].notnull(), 'exchanges_flag'] = 5
#print(df2['exchanges'].notnull())
#df2.head()
df2.exchanges_flag.value_counts()

df2['places'] = df2['places'].replace('',np.nan,regex=True)
df2['places'].isnull().sum()

places_target= []
for x in range(len(df2.places)):
    places_target.append([df2.places[x]])
#print(places_target)
df2['places_target'] = places_target
#df2.head()

train = df2.query('split=="TRAIN"')
test = df2.query('split=="TEST"')
unused = df2.query('split=="NOT-USED"')

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix 
import nltk
from nltk.corpus import stopwords 
stopWords = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer(norm="l2",stop_words='english',max_features=2500) 
train_matrix = tfid.fit_transform(train['text'])
train_matrix = train_matrix.toarray()
test_matrix = tfid.transform(test['text'])
test_matrix = test_matrix.toarray()
#print train_matrix
#print test_matrix
#print tfid

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_matrix, train.places) 
pred = knn.fit(train_matrix, train.places).predict(test_matrix)

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix 
confusion_matrix(test.places, pred)
accuracy_score(test.places, pred) 
