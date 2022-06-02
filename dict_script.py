import pickle
import pipeline
from config import *
import numpy as np

# Read feature vectorizer
f = open(TFID_PATH, "rb")
vectorizer = pickle.load(f)
f.close()

f = open("english_dict.txt", "r")
dict = f.read().splitlines()
f.close()

# Generate bigrams 
inputs = {}
for a in dict:
    for b in dict:
        inputs[a + "_" + b] = pipeline.transform_words(a, b, vectorizer)

# Read model so that it does not have to be retrained
f = open(SVC_PATH, "rb")
model = pickle.load(f)
f.close()
bigrams = {}
for key, X in inputs.items():
    bigrams[key] = model.predict(X)

# sort and write rankings to file
f = open(RANKING_PATH, "w")
sort_orders = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
for i in sort_orders:
	f.write(i[0] + " " + str(i[1]) + "\n")
f.close()