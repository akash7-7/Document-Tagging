
import os
os.chdir("C:\\Users\\509861\\Desktop\\Kaggle n Stuff\\Ignite hackathon\\reuters_articles\\data")
import pandas as pd
from bs4 import BeautifulSoup,SoupStrainer
f = open('C:\\Users\\509861\\Downloads\\reutars\\reut2-021.sgm', 'r')
data= f.read()
soup = BeautifulSoup(data)

print soup.prettify()[0:5000]

title = []
for row in soup.find_all('title', class_=False):
    title.append(row.text)

date= []
for row in soup.find_all('date', class_=False):
    date.append(row.text)

topics = []
for row in soup.find_all('topics', class_=False):
    topics.append(row.text)

places = []
for row in soup.find_all('places', class_=False):
    places.append(row.text)
	
people = []
for row in soup.find_all('people', class_=False):
    people.append(row.text)

orgs = []
for row in soup.find_all('orgs', class_=False):
    orgs.append(row.text)

exchanges = []
for row in soup.find_all('exchanges', class_=False):
    exchanges.append(row.text)

companies = []
for row in soup.find_all('companies', class_=False):
    companies.append(row.text)

text = []
for row in soup.find_all('text', class_=False):
    text.append(row.text)

author = []
for row in soup.find_all('author', class_=False):
    author.append(row.text)

dateline = []
for row in soup.find_all('dateline', class_=False):
    dateline.append(row.text)
	
df = pd.DataFrame({'date':date,'topics':topics,'places':places,'people':people,'orgs':orgs,'exchanges':exchanges,'comapanies':companies,'text':text})


#######################################################################################################################################################


