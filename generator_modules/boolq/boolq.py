import numpy as np
import pandas as pd 
import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import spacy
import nltk
import numpy 
# nltk.download('brown')
# nltk.download('stopwords')
# nltk.download('popular')
from nltk.corpus import stopwords
from sense2vec import Sense2Vec
from nltk import FreqDist
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from generator_modules.text_processing_utils import (tokenize_sentences, get_keywords, 
                                                     get_sentences_for_keyword,get_options,
                                                     filter_phrases,get_sentences_for_keyword_,
                                                     get_adjective_keywords,get_key_sentences_tuple,
                                                     generate_false_statement)
from flashtext import KeywordProcessor




class BoolQGenerator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

        self.s2v = Sense2Vec().from_disk('s2v_old')

        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.set_seed(42)
        
    def set_seed(self,seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    
    
    def generate_questions(self,payload):
        text = payload.get("input_text")
        num_question = payload.get("max_questions", 4)
        sentences = tokenize_sentences(text)  
        adjective_keywords = get_adjective_keywords(self.nlp, text,num_question)
        print("extracted adjectives",adjective_keywords)
        keyword_sentence_pair = get_sentences_for_keyword_(adjective_keywords,sentences)
        print("keyword sentence pair",keyword_sentence_pair)
        keyword_sentence_tuple = get_key_sentences_tuple(keyword_sentence_pair)
        print("keyword sentence tuple",keyword_sentence_tuple)
        bool_res = [bool(random.choice([0,1])) for _ in range(num_question)]
        bool_questions = []
        for index, state in enumerate(bool_res):
            if index >= len(keyword_sentence_tuple): break
            key, statement = keyword_sentence_tuple.pop()
            if not state:
                statement = generate_false_statement(statement,key)
            if statement:
                bool_questions.append({"question": statement, "options":["True","False"],"answer":str(state),"key":key})

        return bool_questions
            
                
        