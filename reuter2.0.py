import os

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.
path = "C:\\Users\\509861\\Downloads\\reutars\\raw"
full_file_paths = get_filepaths(path)
	
import pandas as pd
from bs4 import BeautifulSoup,SoupStrainer

cdata = ""
for i in range(len(full_file_paths)):
    f = open(full_file_paths[i], 'r')
    data= f.read()
    print(i)
    cdata= cdata + data

soup = BeautifulSoup(cdata)
	


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
