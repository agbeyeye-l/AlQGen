import numpy as np
import pandas as pd 
import torch
import random
import spacy
import numpy 
from generator_modules.text_processing_utils import (tokenize_sentences, 
                                                     get_sentences_for_keyword_,
                                                     get_adjective_keywords,get_key_sentences_tuple,
                                                     generate_false_statement)

from generator_modules.utils import QuestionType, ErrorMessages
from generator_modules.models import QuestionRequest, Question
from typing import List


class BoolQGenerator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.set_seed(42)
        
    def set_seed(self,seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def generate_questions(self,corpus:QuestionRequest):
        text = corpus.get("text")
        num_question = corpus.get("num_question", 5)
        
        # if no input text provided
        if len(text)<1:
            return ErrorMessages.noInputTextError()
        
        sentences = tokenize_sentences(text)  
        adjective_keywords = get_adjective_keywords(self.nlp, text,num_question)
        print("extracted adjectives",adjective_keywords)
        keyword_sentence_pair = get_sentences_for_keyword_(adjective_keywords,sentences)
        print("keyword sentence pair",keyword_sentence_pair)
        keyword_sentence_tuple = get_key_sentences_tuple(keyword_sentence_pair)
        print("keyword sentence tuple",keyword_sentence_tuple)
        bool_res = [bool(random.choice([0,1])) for _ in range(num_question)]
        
        bool_questions:List[Question] = []
        for index, state in enumerate(bool_res):
            if index >= len(keyword_sentence_tuple): break
            key, statement = keyword_sentence_tuple.pop()
            if not state:
                statement = generate_false_statement(statement,key)
            if statement and "?" not in statement:
                question = Question(question=statement, options=["True","False"],answer=str(state), question_type=QuestionType.BOOLQ)
                bool_questions.append(question.dict())

        return bool_questions
            
                
        