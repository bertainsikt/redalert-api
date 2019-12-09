# POST API for Red Alert project - NLP and Metalearning components
# Insikt Intelligence S.L. 2019

import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
from utilities import load_data, detect_language
from preprocessing import preprocess, Tagger, remove_stopwords
import json
from gensim.models import KeyedVectors
from Embeddings import Embeddings, to_vector_single, to_vector_single_nonzeros
import numpy as np
import os
from analysis import analyze
from probability_terror import probability_terror
from new_terms_no_lang import new_terms
from classifier import classifier
from claslisting import claslisting
from audit import audit

app = Flask(__name__)

emb_dict = {"en": "embedding-EN", "ar": "embedding-AR", "es": "embedding-ES", "ro": "embedding-RO","fr": "embedding-FR"}

@app.route('/vectorize',methods=['POST'])
def make_vectorize():
    try:
        #Load the data
        data = request.get_json()

    except Exception as e:
        raise e

    if data == {}:
        return(bad_request())
    else:
        #Get the text and the language
        try:
            lang = data['lang']
        except:
            try:
                lang=detect_language(data['text'])
                print(lang)        
            except:   
                responses=jsonify("Error in vectorize: language field is missing")
                return responses      
        try:
           text = data['text']
        except:
           responses=jsonify("Error in vectorize: text is missing")
           return responses 
     
        if lang not in ['en','es','ar','ro','fr']:
            responses=jsonify("Language not available. Language must be in ['en','es','ar','ro','fr']")
            return responses
        #Preprocess the text
        print("Vectorize...")

        embeddings = Embeddings(emb_dict[lang])

        processed_text = preprocess(text)
        no_stpw_text = remove_stopwords(processed_text, lang)
        vectorized_tokens=to_vector_single_nonzeros(no_stpw_text, embeddings,len(no_stpw_text))
	
        if len(vectorized_tokens) > 0:
             vectorized_text = np.mean(vectorized_tokens, axis=0)
        else:
             vectorized_text =np.zeros((300,)*1)
             print(vectorized_text)
        
        #Send the response codes
        responses = jsonify(vector=vectorized_text.tolist())
        responses.status_code = 200
        return responses


@app.route('/probability',methods=['POST'])
def make_probability():
    try:
        #Load the data
        data = request.get_json()

    except Exception as e:
        raise e

    if data == {}:
        return(bad_request())
    else:
        #Get the text,language and classifier
        try:
            lang = data['lang']
        except:
            try:
                lang=detect_language(data['text'])
                print(lang)        
            except:   
                responses=jsonify("Error in vectorize: language field is missing")
                return responses     
        try:
           text = data['text']
        except:
           responses=jsonify("Error in probability: text is missing")
           return responses
        
        try:
            cls = data['classifier']
        except:
           responses=jsonify("Error in probability: classifier is missing")
           return responses
        
        if lang not in ['en','es','ar','ro','fr']:
            responses=jsonify("Language not available. Language must be in ['en','es','ar','ro','fr']")
            return responses
            
        
        #Preprocess the text
        print("Computing probability of having content related to "+cls)

        probability = probability_terror(text,lang,cls)
  
        #Send the response codes
        responses = jsonify(probability=probability)
        responses.status_code = 200
        return responses


@app.route('/analyze',methods=['POST'])
def make_analyze():

    try:
        #Load the data
        data = request.get_json()

    except Exception as e:
        raise e

    if data == {}:
        return(bad_request())
    else:

        #Get the text and the language

        try:
            lang = data['lang']
        except:
            try:
                lang=detect_language(data['text'])
                print(lang)        
            except:   
                responses=jsonify("Error in vectorize: language field is missing")
                return responses    
        try:
           text = data['text'] # we assume text is tokenized
        except:
           responses=jsonify("Error in analyze: text is missing")
           return responses

        if lang not in ['en','es','ar','ro','fr']:
            responses=jsonify( message = "Language not available. Language must be in ['en','es','ar','ro','fr']")
            return responses
            
            
        filename = os.path.join(os.path.dirname(__file__), 'models-registry.json')
        registry = load_data(filename)

        analysis = analyze(text, lang, registry)
        #print(analysis[0])
        #Send the response codes
        responses = jsonify(concepts=analysis[0],key_ideas=analysis[1],topics=analysis[2])
        responses.status_code = 200
        return responses


@app.route('/terms',methods=['POST'])
def make_terms():

    try:
        #Load the data
        data = request.get_json()

    except Exception as e:
        raise e

    if data == {}:
        return(bad_request())
    else:

        texts = data['dataset'] # we assume text is tokenized 
          
	#Preprocess the text
        print("Suggesting new terms for search...")  
        terms=new_terms(texts)
	#print(terms)
        #Send the response codes
        responses = jsonify(message="Suggested new terms for search: ",terms= list(terms))
        responses.status_code = 200
        return responses


@app.route('/sento',methods=['POST'])
def make_sento():

    try:
        #Load the data
        data = request.get_json()

    except Exception as e:
        raise e

    if data == {}:
        return(bad_request())
    else:

        #Get the text, language and classifier
        try:
            lang = data['lang']
        except:
            try:
                lang=detect_language(data['text'])
                print(lang)        
            except:   
                responses=jsonify("Error in vectorize: language field is missing")
                return responses     
        try:
           text = data['text']
        except:
           responses=jsonify("Error in sento: text is missing")
           return responses 
        try:
            cls = data['classifier']
        except:
            responses=jsonify("Error in sento: classifier is missing")
            return responses

        if lang not in ['en','es','ar','ro','fr']:
            responses=jsonify("Language not available. Language must be in ['en','es','ar','ro','fr']")
            return responses
            
        
       
	#Preprocess the text
        print("Sento analysis")  


        # Probability
        probability = probability_terror(text,lang,cls)
        print(probability)

        # Analyze
        filename = os.path.join(os.path.dirname(__file__), 'models-registry.json')
        registry = load_data(filename)

        analysis = analyze(text, lang, registry)
        
        data_audit={"auditEventType":"Start task","details":{"sento":"NLP analysis"},"principal":"Analyst"}
        datajson=json.dumps(data_audit)
        results_audit=audit(datajson)


        #Send the response codes
        responses = jsonify(probability=probability,concepts=analysis[0],key_ideas=analysis[1],topics=analysis[2])
        responses.status_code = 200
        return responses

@app.route('/classifier',methods=['POST'])
def make_classifier():
    try:
        #Load the data
        data = request.get_json()

    except Exception as e:
        raise e

    if data == {}:
        return(bad_request("There is no data for the training"))
    else:
        #Get the text and the language
        try:
            lang = data['lang']
        except:
            try:
                lang=detect_language(data['text'])
                print(lang)        
            except:   
                responses=jsonify("Error in vectorize: language field is missing")
                return responses
        try:
            annotated_data = data['annotated_data']
        except:
            responses=jsonify("Error in classifier: annotated data is missing")
            return responses
        try:
            user_id=data['user_id']
        except:
            responses=jsonify("Error in classifier: user_id is missing")
            return responses
        try:           
            case_id=data['case_id']
        except:
            responses=jsonify("Error in classifier: case_id is missing")
            return responses
        try:        
            clas_name=data['clas_name']
        except:
            responses=jsonify("Error in classifier: classifier name is missing")
            return responses

        print(len(annotated_data))
        if len(annotated_data) < 22:
            responses=jsonify( "Training data set should have more than 10 samples per each class")
            return responses	

        if lang not in ['en','es','ar','ro','fr']:
            responses=jsonify("Language not available. Language must be in ['en','es','ar','ro','fr']")
            return responses
            
        
        #Train the new classifier
        print("Training a new classifier from the user's annotated dataset ")

        accuracy=classifier(annotated_data,lang,user_id,case_id,clas_name)
        
        data_audit={"auditEventType":"Start task","details":{"classifier":"Trains a new classifier based on the annotations provided by the user"},"principal":"Analyst"}
        datajson=json.dumps(data_audit)
        results_audit=audit(datajson)

        #Send the response codes
        responses = jsonify(message="Classifier has been saved. Accuracy given in % - calculated using C-10V", accuracy=accuracy)
        responses.status_code = 200
        return responses

@app.route('/claslisting',methods=['POST'])
def make_claslisting():
        user_id=None
        case_id=None
        try:
            #Load the data
            data = request.get_json()

        except Exception as e:
            raise e

        if data == {}:
           return(bad_request())
        else:
           try:
               user_id=data['user_id']
           except:
               responses=jsonify(message="Error in classifiers listing: user_id is missing")
               return responses
           try:
               case_id=data['case_id']
           except:
               responses=jsonify(message="Error in classifiers listing: case_id is missing")
               return responses
        
        available_classifiers=claslisting(user_id,case_id)
        
        data_audit={"auditEventType":"Start task","details":{"claslisting":"Lists the available classifiers"},"principal":"Analyst"}
        datajson=json.dumps(data_audit)
        results_audit=audit(datajson)

        #Send the response codes
        responses = jsonify(available_classifiers=available_classifiers)
        responses.status_code = 200
        return responses
     

@app.route('/my400')
def bad_request(msg=''):
    code = 400
    if msg=='':
        msg = 'Error'
    return msg, code

if __name__ == '__main__':

    #app.run()
    app.run(host='0.0.0.0',port=5000)
