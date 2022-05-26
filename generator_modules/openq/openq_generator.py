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
from generator_modules.text_processing_utils import tokenize_sentences, get_keywords, get_sentences_for_keyword,get_options

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
        extension = ["Explain your answer.", "Why?", "Argue.", "Elaborate on your answer.", "Give a brief explanation."]
        return extension[random.randint(0,len(extension)-1)]

    def beam_search_decoding (self,inp_ids,attn_mask,model,tokenizer):
        beam_output = model.generate(input_ids=inp_ids,
                                        attention_mask=attn_mask,
                                        max_length=256,
                                    num_beams=10,
                                    num_return_sequences=3,
                                    no_repeat_ngram_size=2,
                                    early_stopping=True
                                    )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
                    beam_output]
        return [Question.strip().capitalize() for Question in Questions]

    def generate_questions(self,payload):
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

        output = self.beam_search_decoding (input_ids, attention_masks,self.model,self.tokenizer)
        
        for i in range(len(output)):
            output[i] = f"{output[i]} {self.open_q_extension()}"
            
        if torch.device=='cuda':
            torch.cuda.empty_cache()
        
        final= {}
        final['Text']= text
        final['Count']= num
        final['Boolean Questions']= output
            
        return final
