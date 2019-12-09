# !/usr/bin/env python
# encoding:utf-8

# Text classification for different languages a different domains 
# Insikt Intelligence S.L. 2019

import csv
import pandas as pd
import numpy as np
import os
import traceback
import time
from nltk.corpus import stopwords
import gensim
import sqlite3
import sys
sys.path.append('../') # path to locate the Utilities folder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pickle
from preprocessing import preprocess, Tagger, remove_stopwords
from Embeddings import Embeddings, to_vector_single, to_vector_single_nonzeros


full_name = {'en':'english','es':'spanish','ar':'arabic','ro':'romanian','fr':'french'}



def probability_terror(text,lang,classifier):


#--------------------------------------------------------------------------------------------
#--- DEFINE FILES AND LANGUAGE
#--------------------------------------------------------------------------------------------

	model_path='./data/probability/insikt/'
	

	if (lang=='en'):
		embedding_name='embedding-EN'
	if (lang=='ar'):
		embedding_name='embedding-AR'
	if (lang=='es'):
		embedding_name='embedding-ES'
	if (lang=='ro'):
		embedding_name='embedding-RO'
	if (lang=='fr'):
                embedding_name='embedding-FR'
		
	if(classifier=='all') and (lang =='en'):
		model_file_JIH='Jihadist-English-Insikt.model'
		model_file_EXR='ExtremeRight-English-Insikt.model'
		sol=two_classifier(text,lang,embedding_name,model_path,model_file_JIH,model_file_EXR)
		return sol

	if(classifier=='all') and (lang =='es'):
                model_file_JIH='Jihadist-Spanish-Insikt.model'
                model_file_EXR='ExtremeRight-Spanish-Insikt.model'
                sol=two_classifier(text,lang,embedding_name,model_path,model_file_JIH,model_file_EXR)
                return sol

	if(classifier=='all') and (lang =='fr'):
                model_file_JIH='Jihadist-French-Insikt.model'
                model_file_EXR='ExtremeRight-French-Insikt.model'
                sol=two_classifier(text,lang,embedding_name,model_path,model_file_JIH,model_file_EXR)
                return sol
	
	if(classifier=='all') and (lang =='ro'):
                model_file_JIH='Jihadist-Romanian-Insikt.model'
                model_file_EXR='ExtremeRight-Romanian-Insikt.model'
                sol=two_classifier(text,lang,embedding_name,model_path,model_file_JIH,model_file_EXR)
                return sol

	if(classifier=='all') and (lang =='ar'):
		model_file_JIH='Jihadist-Arabic-Insikt.model'
		model_file_EXR='ExtremeRight-Arabic-Insikt.model'
		sol=two_classifier(text,lang,embedding_name,model_path,model_file_JIH,model_file_EXR)
		return sol

	if(classifier!='all'):
		sol=one_classifier(text,lang,embedding_name,model_path,classifier)
		return sol

	
		


def one_classifier(text,lang,embedding_name,model_path,model_file):
	
#--------------------------------------------------------------------------------------------
#--- LOAD MODEL AND EMBEDDING
#--------------------------------------------------------------------------------------------
	print(model_file)
	cls=pickle.load(open(model_path+model_file,'rb'))

	embedding = Embeddings(embedding_name)

#--------------------------------------------------------------------------------------------
#--- PROCESSING 
#--------------------------------------------------------------------------------------------

	processed_text = preprocess(text)
	
	no_stpw_text = remove_stopwords(processed_text, lang)
	
	if len(to_vector_single_nonzeros(no_stpw_text,embedding,len(no_stpw_text))) > 0:
		vectorized_text=np.mean(to_vector_single_nonzeros(no_stpw_text,embedding,len(no_stpw_text)),axis=0)
		vectorized_text2=np.reshape(vectorized_text,(1,-1))
		prob=cls.predict_proba(vectorized_text2)[:,1]			
	else:
		vectorized_text=np.zeros((300,)*1)
		prob=0
	#print(cls.classes_) # check that class at second position is L1
	
	for i in list(prob):
		
		return(i)
	


def two_classifier(text,lang,embedding_name,model_path,model_file_JIH,model_file_EXR):
#--------------------------------------------------------------------------------------------
#--- LOAD MODEL AND EMBEDDING
#--------------------------------------------------------------------------------------------


	cls_JIH=pickle.load(open(model_path+model_file_JIH,'rb'))
	cls_EXR=pickle.load(open(model_path+model_file_EXR,'rb'))

	embedding = Embeddings(embedding_name)

#--------------------------------------------------------------------------------------------
#--- PROCESSING 
#--------------------------------------------------------------------------------------------

	processed_text = preprocess(text)
	no_stpw_text = remove_stopwords(processed_text, lang)
	if len(to_vector_single_nonzeros(no_stpw_text,embedding,len(no_stpw_text))) > 0:
		vectorized_text=np.mean(to_vector_single_nonzeros(no_stpw_text,embedding,len(no_stpw_text)),axis=0)
		vectorized_text2=np.reshape(vectorized_text,(1,-1))
		prob_JIH=cls_JIH.predict_proba(vectorized_text2)[:,1]
		prob_EXR=cls_EXR.predict_proba(vectorized_text2)[:,1]
			
	else:
		vectorized_text=np.zeros((300,)*1)
		prob_JIH=0
		prob_EXR=0	

	if prob_JIH > prob_EXR:
		prob=prob_JIH
	else:
		prob=prob_EXR


	for i in list(prob):
		return(i)
