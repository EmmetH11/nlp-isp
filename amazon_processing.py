from config import PUNC
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet, stopwords
import enchant
from autocorrect import Speller

# Remove short words, clean up punctuation
def clean_review(text, lemmatizer, stopwords, autocorrect):
    if not text:
        text = ''
    tokens = word_tokenize(text)
    final_words = []
    for word in tokens:
        word = autocorrect(word.lower())
        if not word in stopwords and word.isalpha():
            for c in word:
                if c in PUNC:
                    word = word.replace(c, "")
            if len(word) >= 3:
                final_words.append(lemma(word, lemmatizer))
    return final_words

def lemma(word, lemmatizer):
    return lemmatizer.lemmatize(word, pos=wordnet.VERB)

def clean_dataset(data, depth):
    lemmatizer = WordNetLemmatizer()
    tfidf_vectorizer = TfidfVectorizer()
    spell = Speller(lang='en')
    stop_words = set(stopwords.words("english"))
    reviews = [clean_review(r["reviewText"], lemmatizer, stop_words, spell) for r in data[:depth]]
    bag_of_words = tfidf_vectorizer.fit_transform([" ".join(r) for r in reviews])
    feature_names = tfidf_vectorizer.get_feature_names()

    labels = [r["overall"] for r in data[:depth]]
    features = pd.DataFrame(bag_of_words.toarray(), columns = feature_names)

    dict = enchant.Dict("en_US")
    for word in feature_names:
        if not dict.check(word):
            features = features.drop(word, axis = 1)

    return features, labels


