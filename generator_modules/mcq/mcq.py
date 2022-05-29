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
from generator_modules.utils import QuestionType, ErrorMessages
from generator_modules.models import QuestionRequest, Question
from typing import List



class MCQGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('Parth/result').to(self.device)
        self.answer_verifier_model = T5ForConditionalGeneration.from_pretrained('Parth/boolean').to(self.device)
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
    
    
    # def build_question_objects(self,model_output,answers):
    #     """form questions"""
        
    #     question_list: List[Question] =[]
    #     for index, val in enumerate(answers):
    #         # get mcq options/distractors
    #         options = get_options(val, self.s2v)
    #         if len(options)<1:
    #             continue
    #         output = model_output[index, :]
    #         decoded_data = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)           
    #         # get question statement
    #         question_text = decoded_data.replace("question:", "").strip()
    #         # filter options and return the best distractors
    #         options = filter_phrases(options, 10,self.normalized_levenshtein) 
    #         options = options if len(options)<7 else options[:6]
    #         question = Question(question=question_text, answer= val, options= options, question_type=QuestionType.MCQ)
    #         question_list.append(question.dict())

    #     return question_list
    
    def formulate_questions(self,answer_question_pair):
        question_list=[]
        print("formulating questions")
        for answer_question in answer_question_pair:
            #get mcq options/distractors
            print("getting distractors")
            options = get_options(answer_question[0], self.s2v)
            if len(options)<1:
                continue
            print("filtering options")
            options = filter_phrases(options, 10,self.normalized_levenshtein) 
            options = options if len(options)<7 else options[:6]
            print("building a question object")
            question = Question(question=answer_question[1], answer= answer_question[0], options= options, question_type=QuestionType.MCQ)
            question_list.append(question.dict())
        return question_list
            
                
    def verify_answer(self,question,context):
        input = f"question: {question} context: {context} </s>"
        print("encoding to verify answer")
        encoded_data = self.tokenizer.encode_plus(input, return_tensors='pt')
        input_ids, attention_masks = encoded_data["input_ids"].to(self.device), encoded_data["attention_mask"].to(self.device)
        print("generating answer by answer verifier model")
        output = self.answer_verifier_model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=256)
        print("decoding verified answer")
        answer =  self.tokenizer.decode(output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        answer = answer.strip().capitalize()
        print("question:", question)
        print("verified answer",answer)
        return answer
        
        
    def generate(self, keyword_sent_mapping, textComponent):
        batch_text = []
        answers = keyword_sent_mapping.keys()
        for answer in answers:
            context = keyword_sent_mapping[answer]
            text = f"context: {context} answer: {answer} </s>"
            batch_text.append(text)

        encoded_data = self.tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors='pt')

        input_ids, attention_masks = encoded_data["input_ids"].to(self.device), encoded_data["attention_mask"].to(self.device)
        print("generating questions")

        with torch.no_grad():
            model_output = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=150)
        print("decoding questions")
        questions_generated = self.tokenizer.batch_decode(model_output,skip_special_tokens=True,clean_up_tokenization_spaces=True)
        print("generated questions are:", questions_generated)
        answer_question_pair=[]
        for index, answer in enumerate(answers):
            context = keyword_sent_mapping[answer]
            
            question = questions_generated[index].replace("question:", "").strip()
         
            if question and answer:
                print("verifying answer")
                verified_answer = str(self.verify_answer(question, textComponent))
                answer_length = len(verified_answer.split(" "))
                print("context:",context)
                print("question:",question)
                print("answer:",answer)
                print("verified answer:",verified_answer)
                print("--------------------------------------------------------")
                if answer_length> 0 and answer_length < 5:
                    print("question and answer has been added")
                    answer_question_pair.append((verified_answer, question))
        # form questions
        #results = self.build_question_objects(model_output,answers)
        results = self.formulate_questions(answer_question_pair)
        return results

                      
    def generate_questions(self, corpus:QuestionRequest):
        text = corpus.get('text')
        num_of_questions = corpus.get('num_question')
        
        # if no input text provided
        if len(text)<1:
            return ErrorMessages.noInputTextError()
        
        questions: List[Question]=[]
        # tokenize text
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
                print("generating mcq questions")
                questions = self.generate(keyword_sentence_mapping,text)
            except Exception as ex:
                # when execption occurs, return 
                print("exception occured so we're returning empty list of questions",ex)
                return questions

            # empty the cudo cache
            if torch.device=='cuda':
                torch.cuda.empty_cache()
                
            return questions
        