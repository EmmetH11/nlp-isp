from autocorrect import Speller
from sklearn.feature_extraction.text import TfidfVectorizer
from config import PUNC

# digest raw text
def digest(body, vectorizer):
    for c in PUNC:
        body = body.replace(c, "")
    spell = Speller(lang='en')
    words = body.split(" ")
    bigrams = []
    for i in range(0, len(words) - 1):
        a = spell(words[i])
        b = spell(words[i+1])
        bigram = a + "_" + b
        if not any(str.isdigit(c) for c in bigram):
            bigrams.append(bigram)
    return vectorizer.transform([" ".join(bigrams)])

def transform_words(a, b, vectorizer):
    return vectorizer.transform([a + "_" + b])
    