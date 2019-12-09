
# Manage pre-trained embeddings
# Insikt Intelligence S.L. 2019

import numpy as np
import os
import os.path
import json
import lmdb
import pickle
import sys
import re

from tqdm import tqdm


# for fasttext binary embeddings
fasttext_support = True
try:
    import fastText
except ImportError as e:
    fasttext_support = False


# gensim is used to exploit .bin FastText embeddings, in particular the OOV with the provided ngrams
#from gensim.models import FastText

# this is the default init size of a lmdb database for embeddings
# based on https://github.com/kermitt2/nerd/blob/master/src/main/java/com/scienceminer/nerd/kb/db/KBDatabase.java

map_size = 100 * 1024 * 1024 * 1024

class Embeddings(object):

    def __init__(self, name, path='./models-registry.json', lang='en', extension='vec'):
        self.name = name
        self.embed_size = 0
        self.static_embed_size = 0
        self.vocab_size = 0
        self.model = {}
        self.registry = self._load_embedding_registry(path)
        self.lang = lang
        self.extension = extension
        self.embedding_lmdb_path = None
        if self.registry is not None:
            self.embedding_lmdb_path = self.registry["embedding-lmdb-path"]
        self.env = None
        self.make_embeddings_simple(name)
        self.static_embed_size = self.embed_size
        self.bilm = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _load_embedding_registry(self, path='./models-registry.json'):
        """
        Load the description of available embeddings. Each description provides a name, 
        a file path (used only if necessary) and a embeddings type (to take into account
        small variation of format)
        """
        registry_json = open(path).read()
        return json.loads(registry_json)


    def make_embeddings_lmdb(self, name="fasttext-crawl", hasHeader=True):
        nbWords = 0
        print('\nCompiling embeddings... (this is done only one time per embeddings at first launch)')
        begin = True
        description = self._get_description(name)
        if description is not None:
            embeddings_path = description["path"]
            embeddings_type = description["type"]
            self.lang = description["lang"]
            print("path:", embeddings_path)
            if embeddings_type == "glove":
                hasHeader = False
            txn = self.env.begin(write=True)
            batch_size = 1024
            i = 0
            nb_lines = 0
            with open(embeddings_path, encoding='utf8') as f:
                for line in f:
                    nb_lines += 1

            with open(embeddings_path, encoding='utf8') as f:
                for line in tqdm(f, total=nb_lines):
                    line = line.split(' ')
                    if begin:
                        if hasHeader:
                            # first line gives the nb of words and the embedding size
                            nbWords = int(line[0])
                            self.embed_size = int(line[1].replace("\n", ""))
                            begin = False
                            continue
                        else:
                            begin = False
                    word = line[0]
                    #if embeddings_type == 'glove':
                    try:
                        if line[len(line)-1] == '\n':
                            vector = np.array([float(val) for val in line[1:len(line)-1]], dtype='float32')
                        else:
                            vector = np.array([float(val) for val in line[1:len(line)]], dtype='float32')
                    
                        #vector = np.array([float(val) for val in line[1:len(line)]], dtype='float32')
                    except:
                        print(len(line))
                        print(line[1:len(line)])
                    #else:
                    #    vector = np.array([float(val) for val in line[1:len(line)-1]], dtype='float32')
                    if self.embed_size == 0:
                        self.embed_size = len(vector)

                    if len(word.encode(encoding='UTF-8')) < self.env.max_key_size():   
                        txn.put(word.encode(encoding='UTF-8'), _serialize_pickle(vector))  
                        i += 1

                    # commit batch
                    if i % batch_size == 0:
                        txn.commit()
                        txn = self.env.begin(write=True)

            #if i % batch_size != 0:
            txn.commit()   
            if nbWords == 0:
                nbWords = i
            self.vocab_size = nbWords
            print('embeddings loaded for', nbWords, "words and", self.embed_size, "dimensions")

    def make_embeddings_simple(self, name="fasttext-crawl", hasHeader=True):
        description = self._get_description(name)
        if description is not None:
            self.extension = description["format"]

        if self.extension == "bin":
            if fasttext_support == True:
                print("embeddings are of .bin format, so they will be loaded in memory...")
                self.make_embeddings_simple_in_memory(name, hasHeader)
            else:
                if not (sys.platform == 'linux' or sys.platform == 'darwin'):
                    raise ValueError('FastText .bin format not supported for your platform')
                else:
                    raise ValueError('Go to the documentation to get more information on how to install FastText .bin support')

        elif self.embedding_lmdb_path is None or self.embedding_lmdb_path == "None":
            print("embedding_lmdb_path is not specified in the embeddings registry, so the embeddings will be loaded in memory...")
            self.make_embeddings_simple_in_memory(name, hashsHeader)
        else:
            # check if the lmdb database exists
            envFilePath = os.path.join(self.embedding_lmdb_path, name)
          
            if os.path.isdir(envFilePath):
                description = self._get_description(name)
                if description is not None:
                    self.lang = description["lang"]

                # open the database in read mode
                self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=4)
                # we need to set self.embed_size and self.vocab_size
                with self.env.begin() as txn:
                    stats = txn.stat()
                    size = stats['entries']
                    self.vocab_size = size

                with self.env.begin() as txn:
                    cursor = txn.cursor()
                    for key, value in cursor:
                        vector = _deserialize_pickle(value)
                        self.embed_size = vector.shape[0]
                        break
                    cursor.close()

                # no idea why, but we need to close and reopen the environment to avoid
                # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
                # when opening new transaction !
                self.env.close()
                self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=2)
            else: 
                # create and load the database in write mode
                self.env = lmdb.open(envFilePath, map_size=map_size)
                self.make_embeddings_lmdb(name, hasHeader)


    def _get_description(self, name):
        for emb in self.registry["embeddings"]:
            if emb["name"] == name:
                return emb
        return None

    def get_word_vector(self, word):
        """
            Get static embeddings (e.g. glove) for a given token
        """
        if (self.name == 'wiki.fr') or (self.name == 'wiki.fr.bin'):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if self.env is None or self.extension == 'bin':
            # db not available or embeddings in bin format, the embeddings should be available in memory (normally!)
            return self.get_word_vector_in_memory(word)
        try:  
            with self.env.begin() as txn:
                txn = self.env.begin()   
                vector = txn.get(word.encode(encoding='UTF-8'))
                #print('get_word_vector'+str(vector))
                if vector:
                    word_vector = _deserialize_pickle(vector)
                    vector = None
                else:
                    word_vector = np.zeros((self.static_embed_size,), dtype=np.float32)
                    # alternatively, initialize with random negative values
                    #word_vector = np.random.uniform(low=-0.5, high=0.0, size=(self.embed_size,))
                    # alternatively use fasttext OOV ngram possibilities (if ngram available)
        except lmdb.Error:
            # no idea why, but we need to close and reopen the environment to avoid
            # mdb_txn_begin: MDB_BAD_RSLOT: Invalid reuse of reader locktable slot
            # when opening new transaction !
            self.env.close()
            envFilePath = os.path.join(self.embedding_lmdb_path, self.name)
            self.env = lmdb.open(envFilePath, readonly=True, max_readers=2048, max_spare_txns=2, lock=False)
            return self.get_word_vector(word)
        return word_vector

    def vectorize_sentence(self, sentence): 
        return np.mean([self.get_word_vector(word) for word in sentence if word in self.model] or [np.zeros(self.embed_size)], axis=0)

    def get_word_vector_in_memory(self, word):
        if (self.name == 'wiki.fr') or (self.name == 'wiki.fr.bin'):
            # the pre-trained embeddings are not cased
            word = word.lower()
        if self.extension == 'bin':
            return self.model.get_word_vector(word)
        if word in self.model:
            return self.model[word]
        # for unknown word, we use a vector filled with 0.0
        return np.zeros((self.static_embed_size,), dtype=np.float32)
        # alternatively, initialize with random negative values
        #return np.random.uniform(low=-0.5, high=0.0, size=(self.embed_size,))
        # alternatively use fasttext OOV ngram possibilities (if ngram available)


def _serialize_byteio(array):
    import io
    memfile = io.BytesIO()
    np.save(memfile, array)
    memfile.seek(0)
    return memfile.getvalue()


def _deserialize_byteio(serialized):
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    return np.load(memfile)


def _serialize_pickle(a):
    return pickle.dumps(a)


def _deserialize_pickle(serialized):
    return pickle.loads(serialized)

def to_vector_single(tokens, embeddings, maxlen=300, lowercase=False, num_norm=True):
	"""
    Given a list of tokens convert it to a sequence of word embedding 
    vectors with the provided embeddings, introducing <PAD> and <UNK> padding token
    vector when appropriate
	"""
	window = tokens[-maxlen:]
	
    # TBD: use better initializers (uniform, etc.) 
	x = np.zeros((maxlen, embeddings.embed_size), )

    # TBD: padding should be left and which vector do we use for padding? 
    # and what about masking padding later for RNN?
	for i, word in enumerate(window):
		if lowercase:
			word = _lower(word)
			
		if num_norm:
			word = _normalize_num(word)
			#print('to vector single'+ word)
		x[i,:] = embeddings.get_word_vector(word).astype('float32')
		

	return x

def to_vector_single_nonzeros(tokens, embeddings, maxlen=300, lowercase=False, num_norm=True):


	return np.array([embeddings.get_word_vector(word).astype('float32') for word in tokens if embeddings.get_word_vector(word).astype('float32').all() !=np.zeros(embeddings.embed_size).all()])
				




def _lower(word):
    return word.lower() 


def _normalize_num(word):
    return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
def test():
    embeddings = Embeddings("glove-840B")
    token_list = [['This', 'is', 'a', 'test', '.']]
    vect = embeddings.get_sentence_vector_ELMo(token_list)
    embeddings.cache_ELMo_lmdb_vector(token_list, vect)
    vect = embeddings.get_sentence_vector_ELMo(token_list)

    embeddings.clean_ELMo_cache()
