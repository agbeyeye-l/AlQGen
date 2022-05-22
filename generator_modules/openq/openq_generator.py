import numpy as np
import pandas as pd 
import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import spacy
import nltk
import numpy 
nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import stopwords
from sense2vec import Sense2Vec
from nltk import FreqDist
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from generator_modules.text_processing_utils import tokenize_sentences, get_keywords, get_sentences_for_keyword,get_options,filter_phrases
from generator_modules.encoding.encoding import beam_search_decoding

class OpenQGenerator:
       
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions').to(self.device)
        self.set_seed(42)
        
    def set_seed(self,seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def random_choice(self):
        a = random.choice([0,1])
        return bool(a)
    
    def open_q_extension(self):
        extension = ["Explain your answer.", "Why?", "Argue."]
        return extension[random.randint(0,len(extension)-1)]

    def generate_question(self,payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }

        text = inp['input_text']
        num= inp['max_questions']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)
        answer = self.random_choice()
        form = "truefalse: %s passage: %s </s>" % (modified_text, answer)

        encoding = self.tokenizer.encode_plus(form, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        output = beam_search_decoding (input_ids, attention_masks,self.model,self.tokenizer)
        
        for i in range(len(output)):
            output[i] = f"{output[i]} {self.open_q_extension()}"
            
        if torch.device=='cuda':
            torch.cuda.empty_cache()
        
        final= {}
        final['Text']= text
        final['Count']= num
        final['Boolean Questions']= output
            
        return final
