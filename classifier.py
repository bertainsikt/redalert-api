# !/usr/bin/env python
# encoding:utf-8

# Trains a new text classifier based on supervised learning and calculates its cv-10-fold accuracy
# Insikt Intelligence S.L. 2019

import pandas as pd
import numpy as np
import os
import gensim
import sqlite3
import sys
sys.path.append('../') # path to locate the Utilities folder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pickle
from preprocessing import preprocess, Tagger, remove_stopwords
from Embeddings import Embeddings, to_vector_single, to_vector_single_nonzeros

full_name = {'en':'english','es':'spanish','ar':'arabic','ro':'romanian','fr':'french'}

def classifier(annotated_data,lang,user_id,case_id,clas_name):


#--------------------------------------------------------------------------------------------
#--- DEFINE FILES AND LANGUAGE
#--------------------------------------------------------------------------------------------

	model_path='./data/probability/insikt/'
	model_file=user_id+'_'+case_id+'_'+clas_name+'_classifier.model'

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
	
	embedding = Embeddings(embedding_name)
#--------------------------------------------------------------------------------------------
#--- GENERAL SCRIPT
#--------------------------------------------------------------------------------------------

########## Tokenize + stopwords
	#print(annotated_data)
	#raw_data=np.array(annotated_data)
	x_train= [i[0] for i in annotated_data]
	#print(x_train)	
	y_train = [i[1] for i in annotated_data] #replace N0 for L0...!!!
	#print(y_train)
	x_train_DL=[]
	
	print('Data training with '+str(len(x_train))+' texts')

	for text in x_train:
		#print(text)
		processed_text = preprocess(text)
		no_stpw_text = remove_stopwords(processed_text, lang)
		if len(to_vector_single_nonzeros(no_stpw_text,embedding,len(no_stpw_text))) > 0:
			vectorized_text=np.mean(to_vector_single_nonzeros(no_stpw_text,embedding,len(no_stpw_text)),axis=0)
		else:
			vectorized_text=np.zeros((300,)*1)
		#print(vectorized_text)
		#x_train_DL.append(np.reshape(vectorized_text,(1,-1)))
		x_train_DL.append(vectorized_text)

########## Build and test classifiers with 10-fold -cross validation

	skf = StratifiedKFold(n_splits=10,shuffle=True)

#	Stochastic Descent Gradient
		
	cls = SGDClassifier(loss="log", penalty="l2", max_iter=500).fit(x_train_DL,y_train)
	scores = cross_val_score(cls,x_train_DL,y_train,cv=skf,scoring='accuracy')
	print("Accuracy C-10V EN: %2.1f (+/- %2.1f)" % (100*scores.mean(), scores.std() * 200))
	print(cls.classes_) # check that class at the second position is 'Yes'
	accuracy=round((100*scores.mean()),2)
########## Save the model

	pickle.dump(cls,open(model_path+model_file,'wb'))
	return(accuracy)

	


