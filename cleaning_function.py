import re
def text_to_words( raw_text ):        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
    words = letters_only.lower().split()                             
    return( " ".join(words))
