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
import utils

class MCQGenerator:
    def __init__(self):
        
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('Parth/result')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.device = device
        self.model = model
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
    
    def build_question_objects(self,model_output,answers):
        # Build questions
        question_list =[]
        for index, val in enumerate(answers):
            # get mcq options/distractors
            options = get_options(val, self.s2v)
            if len(options)<1:
                continue
            question_object ={}
            output = model_output[index, :]
            decoded_data = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)           
            # get question statement
            question_object["question"] = decoded_data.replace("question:", "").strip()
            question_object["question_type"] = utils.QuestionType.MCQ
            question_object["answer"] = val
            # filter options and return the best distractors
            options = filter_phrases(options, 10,self.normalized_levenshtein) 
            question_object["options"] = options if len(options)<7 else options[:6]

            question_list.append(question_object)

        return question_list
        
        
    def generate(self, keyword_sent_mapping):
        batch_text = []
        answers = keyword_sent_mapping.keys()
        for answer in answers:
            context = keyword_sent_mapping[answer]
            text = f"context: {context} answer: {answer} </s>"
            batch_text.append(text)

        encoding = self.tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")

        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outs = self.model.generate(input_ids=input_ids,
                                attention_mask=attention_masks,
                                max_length=150)
        # form questions
        results = self.build_question_objects(outs)
        return results

        
                
    def generate_questions(self, corpus):
        questions = {}
        text = corpus.get("input_text","")
        num_of_questions = corpus.get("max_questions", 5)
        
        sentences = tokenize_sentences(text)
        text = " ".join(sentences)
        
        # extract keywords
        keywords = get_keywords(self.nlp,text,num_of_questions,self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )
        # map keywords to sentences
        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

        # if no key sentence map is found return empty list
        if len(keyword_sentence_mapping.keys()) == 0:
            return questions
        else:
            try:
                questions = self.generate(keyword_sentence_mapping)
            except Exception as ex:
                # when execption occurs, return 
                return questions

            # empty the cudo cache
            if torch.device=='cuda':
                torch.cuda.empty_cache()
                
            return questions
        