# Python utilities used in Red Alert API
# Insikt Intelligence S.L. 2019

import functools
import json
from textblob import TextBlob
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

def _search(forward, source, target, start=0, end=None):
    """Naive search for target in source."""
    m = len(source)
    n = len(target)
    if end is None:
        end = m
    else:
        end = min(end, m)
    if n == 0 or (end-start) < n:
        # target is empty, or longer than source, so obviously can't be found.
        return None
    if forward:
        x = range(start, end-n+1)
    else:
        x = range(end-n, start-1, -1)
    for i in x:
        if source[i:i+n] == target:
            return i
    return None

search = functools.partial(_search, True)
rsearch = functools.partial(_search, False)

def load_data(path):
    """
    Load data stored as json. 
    """
    data_json = open(path).read()
    return json.loads(data_json)

def detect_language(text):
    input = TextBlob(text)
    language= input.detect_language()
    return language

def balance(data):
        zeroes = data.loc[data['score'] == 0]
        non_zeroes = data.loc[data['score'] > 0]
        ones= data.loc[data['score'] == 1]
        small_zeroes =  zeroes.sort_values('date', ascending=False)[:len(ones)]
        data_balanced = pd.concat([small_zeroes,non_zeroes])
        return data_balanced

def preprocess(texts,lan):
        processed = []
        stopwords_list = set(stopwords.words(lan))
        new_stopwords=['http','https','https:','https://','saw','vimeo','youtube','someone','is','an','he','him','you','the','them','to','th','and','it','xlyssxmaria']
        #with open('/home/users/berta/nltk_data/corpora/stopwords/english') as f:
        #       lines = f.read().splitlines()
        #       new_stopwords.extend(lines)
        new_stopwords_list=stopwords_list.union(new_stopwords)

        not_stopwords=['no','not']
        final_stopwords_list=set([word for word in new_stopwords_list if word not in not_stopwords])
        for tweet in texts:
                words = []
                if not type(tweet).__name__=='str':
                        print(type(tweet))
                        print(tweet)
                for w in tweet.split():
                        n = ''.join(c for c in w.lower())

                        if n not in final_stopwords_list and len(n)>0:
                                words.append(n)
                                #print(n)
                processed.append(words)
        return processed


def vectorize(model, texts):
        embeddings_dim = 300
        x = np.zeros((len(texts), embeddings_dim), )

        for i,w in enumerate(texts):
                if w in model:
                        x[i,:]=model[w]


        vector_mean=np.mean(x,axis=0)
        result=np.reshape(vector_mean,(1,-1))

        return result

