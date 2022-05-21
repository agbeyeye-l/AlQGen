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
    
    def generate_questions_mcq(keyword_sentence_mapping):
        batch_text = []
        answers = keyword_sent_mapping.keys()
        for answer in answers:
            txt = keyword_sent_mapping[answer]
            context = "context: " + txt
            text = context + " " + "answer: " + answer + " </s>"
            batch_text.append(text)

        encoding = self.tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")


        print ("Running model for generation")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outs = self.model.generate(input_ids=input_ids,
                                attention_mask=attention_masks,
                                max_length=150)

        output_array ={}
        output_array["questions"] =[]
    #     print(outs)
        for index, val in enumerate(answers):
            individual_question ={}
            out = outs[index, :]
            dec = self.tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            Question = dec.replace("question:", "")
            Question = Question.strip()
            individual_question["question_statement"] = Question
            individual_question["question_type"] = "MCQ"
            individual_question["answer"] = val
            individual_question["id"] = index+1
            individual_question["options"], individual_question["options_algorithm"] = get_options(val, self.s2v)

            individual_question["options"] =  filter_phrases(individual_question["options"], 10,self.normalized_levenshtein)
            index = 3
            individual_question["extra_options"]= individual_question["options"][index:]
            individual_question["options"] = individual_question["options"][:index]
            individual_question["context"] = keyword_sent_mapping[val]
        
            if len(individual_question["options"])>0:
                output_array["questions"].append(individual_question)

        return output_array

        
                
    def predict_mcq(self, payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)


        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )


        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)

        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

   
        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            return final_output
        else:
            try:
                generated_questions = self.generate_questions_mcq(keyword_sentence_mapping)

            except:
                return final_output
            end = time.time()

            final_output["statement"] = modified_text
            final_output["questions"] = generated_questions["questions"]
            final_output["time_taken"] = end-start
            
            if torch.device=='cuda':
                torch.cuda.empty_cache()
                
            return final_output
        