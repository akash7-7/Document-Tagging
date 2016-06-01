################STRUCTURING########################################

import os
import re
import pandas as pd
from bs4 import BeautifulSoup,SoupStrainer

def get_filepaths(directory):
	file_paths = [] 
	for root, directories, files in os.walk(directory):
		for filename in files:
			
			filepath = os.path.join(root, filename)
			file_paths.append(filepath)  

	return file_paths  

path = "C:\\Users\\509861\\Downloads\\reutars\\raw"
full_file_paths = get_filepaths(path)
				
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

##############PUTTING IT TOGETHER#####################################
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
from sklearn.linear_model import SGDClassifier

from textblob import TextBlob

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

############CLEANING => NOT WORKING#################################

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import re
stop_words = set(stopwords.words('english'))
stop_words.update(['-','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 

clean_text = []

for i in range(len(df.text)):
	clean_text.append(re.sub(r'\d+','',df.text[i]))
	print(i)

df['clean_text'] = clean_text

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

tfidf = TfidfVectorizer(norm="l2",max_features=100,lowercase=True
					   stop_words = 'english',ngram_range=(1, 3)) 

train_matrix = tfidf.fit_transform(train.clean_text)
train_matrix = train_matrix.todense()
test_matrix = tfidf.transform(test.clean_text)
test_matrix = test_matrix.todense()

#algo options:
one_vs_rest = OneVsRestClassifier(LinearSVC(random_state=0))
knn = KNeighborsClassifier()
NB = MultinomialNB()
rfc = RandomForestClassifier(n_estimators=10)


from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline

#####PIPELINE & CV FUNCTION FOR EACH TARGET###########################


#target options: places_target,people_target,exchanges_target,orgs_target,topics_target

def cv_scores_places(algo,label):
	
	pipeline = Pipeline([
	('tfidf', TfidfVectorizer()), 
	('algo', algo,)
	])


	scores = cross_val_score(pipeline,  # steps to convert raw messages into models
							train.clean_text,  # training data
							train[label],  # training labels
							cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
							scoring='accuracy',  # which scoring metric?
							n_jobs=-1,  # -1 = use all cores = faster
							)
	print scores


#############PREDICTION AND ACCURACY#########################################

import difflib

#algo options:
one_vs_rest = OneVsRestClassifier(LinearSVC(random_state=0))
knn = KNeighborsClassifier()
NB = MultinomialNB()
rfc = RandomForestClassifier(n_estimators=10)
sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)

#label options: places_target,people_target,exchanges_target,orgs_target,topics_target

def pred_accuracy(algo,label):
	X = train_matrix
	y = train[label]
	results= algo.fit(X, y).predict(test_matrix).astype(object)
	target = np.asarray(test[label])
	match = 0
	for i in range(len(test)):
		match = match + difflib.SequenceMatcher(None,target[i],results[i]).ratio()
		print(i)
	print(float(match)/len(test))

####NEED TO TEST########GRID-SEARCH##############################################

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

clf = SVC()
pipeline_svm = Pipeline([
	('tfidf', TfidfVectorizer()),
	('clf', SVC()),  # <== change here
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
	cv=StratifiedKFold(train.places_target, n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(train_matrix, train.places_target) # find the best combination from param_svm
svm_detector.grid_scores_
