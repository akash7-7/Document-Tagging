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
