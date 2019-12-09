# NLP processing utilities
# Insikt Intelligence S.L. 2019

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import numpy as np
from scipy import spatial
from joblib import load
import stanfordnlp

full_lang = {'en':'english', 'es':'spanish', 'ar':'arabic', 'ro':'romanian','fr':'french'}

def preprocess(text):
    #receives raw text
    words = []
    if not type(text).__name__=='str':
        print(type(text))
        print(text)
    for w in text.split():
        n = ''.join(c for c in w.lower())
            
        if len(n)>0:
            words.append(n)
    return words


def remove_stopwords(text, lang):
    # receives tokenized text
    stopwords_list = set(stopwords.words(full_lang[lang]))
    new_stopwords=['http','https','saw','vimeo','youtube','someone','is','an','he','him','you','the','them','to','th','and','it','xlyssxmaria']
    new_stopwords_list=stopwords_list.union(new_stopwords)

    not_stopwords=['no','not']
    final_stopwords_list=set([word for word in new_stopwords_list if word not in not_stopwords])

    words = []
    for w in text:
        if w not in final_stopwords_list:
            words.append(w)         
    return words



ud2penn = {'ADJ':'JJ', 'ADP':'IN', 'ADV':'RB', 'AUX':'VB', 'CCONJ':'CC', 'DET':'DT', 'INTJ':'UH', 
           'NOUN':'NN', 'PART':'RP', 'PRON':'PRP', 'PROPN':'NNP', 'PUNCT':'SYM', 'SCONJ':'IN',
           'SYM':'SYM', 'VERB':'VB', 'X':'SYM'}


class Tagger():

    def __init__(self, lang, path):
        
        self.lang = lang
        self.path = path
        if lang == 'ro':
            self.tagger = load(path['ro_pos'])  

        # if lang == 'ar':
        #     self.tagger = stanfordnlp.Pipeline(models_dir="/home/paula/Workspace/code/NLP/models/stanford", lang='ar', use_gpu=False)



    def pos_tag(self,text):

        token_text = text
        text = ' '.join(text)

        if self.lang=='en':
            return nltk.pos_tag(token_text)

        elif self.lang in ['ro']:

            tags = self.tagger.predict([features(token_text, index) for index in range(len(token_text))])
            return self.change_ud2penn([(token_text[i], tags[i]) for i in range(len(token_text))])
 
        elif self.lang=='ar':

            pipeline = stanfordnlp.Pipeline(models_dir="models/stanford", lang='ar', use_gpu=False, processors='tokenize,pos')
            doc = pipeline(text)
            tags = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]         
            
            return tags
 
        elif self.lang=='es':
            # blob =  Text(text, hint_language_code=self.lang)
            # return self.change_ud2penn(blob.pos_tags)
            pipeline = stanfordnlp.Pipeline(models_dir="models/stanford", lang='es', use_gpu=False, processors='tokenize,pos')
            doc = pipeline(text)
            tags = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]         
            return tags
       
        elif self.lang =='fr':
            return nltk.pos_tag(token_text)

    def change_ud2penn(self, tagged):

        new_tagged = []
        for tag in tagged:
            if tag[1] in ud2penn.keys():
                new_tagged.append((tag[0],ud2penn[tag[1]]))
            else:
                new_tagged.append(tag)

        return new_tagged

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit()
        }
