import numpy as np
import pandas as pd 
import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import numpy 
from generator_modules.text_processing_utils import tokenize_sentences
from generator_modules.utils import QuestionType, ErrorMessages
from generator_modules.models import QuestionRequest, Question
from typing import List


class OpenQGenerator:
       
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('newlife/openq-generator').to(self.device)
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

    def decoder (self,inp_ids,attn_mask,model,tokenizer):
        beam_output = model.generate(input_ids=inp_ids,
                                        attention_mask=attn_mask,
                                        max_length=256,
                                    num_beams=10,
                                    num_return_sequences=3,
                                    no_repeat_ngram_size=2,
                                    early_stopping=True
                                    )
        outputs = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
                    beam_output]
        return [output.strip().capitalize() for output in outputs]

    def generate_questions(self,corpus:QuestionRequest):
        text = corpus.get('text')
        num_questions = corpus.get('num_question')

        # if no input text provided
        if len(text)<1:
            return ErrorMessages.noInputTextError()
        
        question_list: List[Question]=[]
        sentences = tokenize_sentences(text)
        num_open_question = num_questions//3
        partition_texts=[]
        num_sentences_per_partition = len(sentences)//num_open_question
        start, end = 0, num_sentences_per_partition
        for _ in range(num_open_question):
            if len(sentences[start:end])>1:
                partition_texts.append(" ".join(sentences[start:end]))
                start,end = end, end+num_sentences_per_partition
        
        
        answers = [self.random_choice() for _ in range(num_open_question) ]
        for index,answer in enumerate(answers):   
            if index<len(partition_texts) :          
                model_input= f"truefalse: {answer} passage: {partition_texts[index]} </s>"

                encoding = self.tokenizer.encode_plus(model_input, return_tensors="pt")
                input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

                outputs = self.decoder (input_ids, attention_masks,self.model,self.tokenizer)
            
            
                question = Question(question=f"{outputs[0]} {self.open_q_extension()}", options=[], answer='', question_type=QuestionType.OPENQ)
                question_list.append(question.dict())
            
        if torch.device=='cuda':
            torch.cuda.empty_cache()
            
        return question_list
