'''Spam classifier'''

import os
os.chdir("C:\\Users\\509861\\Desktop\\Kaggle n Stuff\\NLP\\spam_classifier")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix

messages = pd.read_csv("SMSSpamCollection",sep="\t",names=['label','message'])
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

tfidf = TfidfVectorizer(norm="l2")

train_data_features = tfidf.fit_transform(msg_train)
train_data_features = train_data_features.todense()

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features,label_train)

test_data_features = tfidf.transform(msg_test)
test_data_features = test_data_features.todense()
results =  forest.predict(test_data_features)

confusion_matrix(label_test, results)
accuracy_score(label_test, results)
