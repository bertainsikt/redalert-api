# !/usr/bin/env python
# encoding:utf-8

# Goal: Suggesting set of terms2 for searching online content given a data set of online content found with set of terms1.
# Insikt Intelligence S.L. 2019

from utilities import search, load_data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Embeddings import Embeddings, to_vector_single
from preprocessing import Tagger, remove_stopwords
import numpy as np
from scipy import spatial
import pandas as pd
import os
from itertools import chain
from nltk import FreqDist

full_lang = {'en':'english', 'es':'spanish', 'ar':'arabic', 'ro':'romanian'}

def preprocess(texts):
	processed = []
	stopwords_list_en = set(stopwords.words((full_lang['en'])))
	stopwords_list_ar = set(stopwords.words((full_lang['ar'])))
	stopwords_list_es = set(stopwords.words((full_lang['es'])))
	stopwords_list_ro = set(stopwords.words((full_lang['ro'])))
	stopsymbols=[',',':','/','|','!','?','¿','¡','-','_','.',';']
	
	new_stopwords_list=stopwords_list_en.union(stopwords_list_es,stopwords_list_ro,stopwords_list_ar,stopsymbols)

	final_stopwords_list=set([word for word in new_stopwords_list])
	for tweet in texts:
		words = []
		if not type(tweet).__name__=='str':
			print(type(tweet))
#			print(tweet)
		for w in tweet.split():
			n = ''.join(c for c in w.lower())
			
			if n not in final_stopwords_list and len(n)>0:
				words.append(n)
				
		processed.append(words)
	return processed


def new_terms(texts):
	#df=pd.read_csv(csv_file)
	#texts=list(df.iloc[:,0])

	tokens = preprocess(texts)
	tokens_join = list(chain.from_iterable(tokens))
	#print('tokens')
	#print(tokens_join)
	fdist=FreqDist(str(w) for w in tokens_join)



	mostCommon=fdist.most_common(100)
	return mostCommon



