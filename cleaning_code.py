#final cleaning code( can  change porter with lemma)

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
import re
stop_words = set(stopwords.words('english'))
stop_words.update(['-','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 

def clean(article):
	article = re.sub(r'\d+','',article)
	for articles in article:
		return [porter.stem(i.lower()) for i in wordpunct_tokenize(article) if i.lower() not in stop_words and len(i)>2]
		
		
###########################################################################################################################
from nltk.corpus import stopwords

stop_words = stopwords.words("english")

def split_into_lemmas(article):
    article = re.sub(r'\d+','',unicode(article, 'utf8'))
    sentences = TextBlob(article).lower()
    words = i for i in sentence.split() if i not in stop_words
    words = TextBlob(article).words
    return [word.lemma for word in words]

def split_into_lemmas(article):
    article = re.sub(r'\d+','',unicode(article, 'utf8'))
    words = TextBlob(article).words.lower()
    meaningful_words = [w for w in words if not w in stops_words] 
    return [meaningful_words.lemma for word in words]

from textblob import TextBlob
from nltk.corpus import stopwords
def article_to_words( article ):
    letters_only = re.sub("[^a-zA-Z]", " ", article) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join(meaningful_words)) 
