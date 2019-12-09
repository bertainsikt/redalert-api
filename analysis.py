# NLP Analsys
# Insikt Intelligence S.L. 2019

from utilities import search, load_data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Embeddings import Embeddings, to_vector_single, to_vector_single_nonzeros
from preprocessing import preprocess, Tagger, remove_stopwords

import numpy as np
from scipy import spatial


full_lang = {'en':'english', 'es':'spanish', 'ar':'arabic', 'ro':'romanian','fr':'french'}
#emb_dict = {"en": "jihadist-en", "ar": "jihadist-ar", "es": "aligned-ES-en", "ro": "wiki-ro"}
emb_dict = {"en": "embedding-EN", "ar": "embedding-AR", "es": "embedding-ES", "ro": "embedding-RO","fr":"embedding-FR"}

def get_concepts(pos, lang):
    wordnet_lemmatizer = WordNetLemmatizer()
    sstopwords = stopwords.words(full_lang[lang])

    concepts = []

    for w in pos:
        if w[0] not in sstopwords and w[1][0] in ['J', 'A', 'N', 'V']:
            concepts.append(wordnet_lemmatizer.lemmatize(w[0]))

    return list(set(concepts))


def get_key_ideas(pos, lang, patterns_path):
    # load the patterns
    patterns = load_data(patterns_path)
    patterns = patterns["default_patterns_"+lang]
    # exceptions = patterns["default_exceptions_"+lang]

    # get the key ideas
    key_ideas = []

    tokens = [w[0] for w in pos]
    words = []
    for w in pos:
        if w[1][0] in ['J', 'V', 'N']:
            words.append(w[1][0])
        else:
            words.append(w[0])
    # patterns = [x[0] for x in self.patterns]
        
    for p in patterns:
        start = search(words, p)

        if start:
            key_idea = tokens[start:start+len(p)]
            key_ideas.append(key_idea)

    return key_ideas


def get_topics(text, lang, topics_path):
    #initialization   
    embeddings = Embeddings(emb_dict[lang])

    # get the topics dictionary from the path
    topics_dicts = load_data(topics_path)
    topics_dict = topics_dicts[lang]
        
    topics = list(topics_dict.keys())

    if lang=='en':
        #cl = 0.7 # when a topic is "close"
        cl=0.5
    else:
        cl = 0.5
    # now vectorize the topics
    vect_dict_topics = [(w, np.mean(to_vector_single_nonzeros(topics_dict[w], embeddings, len(topics_dict[w])), axis=0)) for w in topics]
    #print(vect_dict_topics)

    # get topics
    assigned_topics = []
    dists = []
   
    if len(to_vector_single_nonzeros(text, embeddings, len(text))) > 0:
        vectorized_text = np.mean(to_vector_single_nonzeros(text, embeddings, len(text)), axis=0)
    else:
        vectorized_text =np.zeros((300,)*1)

    for v in vect_dict_topics:
        dists.append(spatial.distance.cosine(vectorized_text, v[1])) # measure distance to all topics

    good_topics = [topics[i].upper() for i in range(len(topics)) if dists[i]<cl] # choose close topics
    if not good_topics:
        good_topics.append('OTHER')

            # assigned_topics.append(topic)
    assigned_topics.append(good_topics)

    return assigned_topics


def analyze(text, lang, registry):

    topics_path = registry['topics']["topics_path"]
    patterns_path = registry["key_ideas"]["patterns_path"]
    
    processed_text = preprocess(text)
    no_stpw_text = remove_stopwords(processed_text, lang)

    tagger = Tagger(lang, registry['pos_models'])
    pos = tagger.pos_tag(processed_text)

    concepts = get_concepts(pos, lang)
    key_ideas = get_key_ideas(pos, lang, patterns_path)
    topics = get_topics(no_stpw_text, lang, topics_path)

    result=[concepts,key_ideas,topics]
    #return {"concepts": concepts, "key_ideas": key_ideas, "topics": topics}
    return result
