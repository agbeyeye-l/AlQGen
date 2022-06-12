from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sense2vec import Sense2Vec
from generator_modules.utils import QuestionType, ErrorMessages
from generator_modules.models import Question, DistractorRequest
from generator_modules.text_processing_utils import get_options, tokenize_sentences
import nltk
nltk.download('wordnet')
import json
from nltk.corpus import wordnet as wn
import requests
from flashtext import KeywordProcessor


class DistractorGenerator:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(self.device)
        self.s2v = Sense2Vec().from_disk('s2v_old')
    
    def get_distractors(self,answer, context):
        distractors=[]
        if len(answer.split(' '))>1:
            print("illegal")
            distractors = sense2vec_get_words(answer,self.s2v)
        if len(distractors)<1:  
            print("here")
            distractors.extend(wn_get_distractors(answer,context))

        print(distractors)
        return distractors 
    
    def get_context(self,answer, text):
        context =[]
        keyword_processor = KeywordProcessor()
        keyword_processor.add_keyword(answer.split()[-1]) 
        sentences = tokenize_sentences(text)
        for sentence in sentences:
            if len(context)>3:
                return ".".join(context[:3])
            found = keyword_processor.extract_keywords(sentence)
            if found:
                context.append(sentence)
        return ".".join(context)
        
    def generate(self,corpus:DistractorRequest):
        text = corpus.get('text')
        questions = corpus.get('questions')
        result=[]
        if len(text)<1:
            return ErrorMessages.noInputTextError()
        
        for question in questions:
            inputs = self.tokenizer(question, text,truncation=True,max_length=512, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]

            outputs = self.model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            # Get the most likely beginning of answer with the argmax of the score
            answer_start = torch.argmax(answer_start_scores)
            # Get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            )
            if "?" not in answer and len(answer)<100:
                try:
                    context = self.get_context(answer,text)
                    options = self.get_distractors(answer, context)
                except Exception as ex:
                    print("an exception occured",ex)
                    options =  []
                question_object = Question(question=question, answer=answer, options=options, question_type=QuestionType.MCQ)
            else:
                question_object = Question(question=question, answer="Not found", options=[], question_type=QuestionType.MCQ)
            result.append(question_object.dict())
            
        return result


from collections import OrderedDict
import random
def sense2vec_get_words(word,s2v):
    output = []
    word = word.lower()
    word = word.replace(" ", "_")

    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=20)
    except:
      return []
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ").lower()
        if append_word.lower() != word:
            output.append(append_word.title())

    out = list(OrderedDict.fromkeys(output))
    out = list(set(out))
    random.shuffle(out)
    return out[:6]

def process_word(word):
  tokens = word.split(' ')
  if len(tokens)>1:
    return tokens[-1], " ".join(tokens[:len(tokens)-1])
  return word, ""

WSD_URL = "https://wsd-hwjogbqlea-ey.a.run.app/getsense"
def wn_get_distractors(word, context):   
    word, reserve = process_word(word)
    res = []
    try:
        response = requests.post(WSD_URL, data=json.dumps({"context":context, "keyword":word}))
        result = response.json()
    except:
        return res
    sense = result.get('sense')
    print("defin result",result.get('definition'))
    hyponyms = wn.synset(sense).hypernyms()[0].hyponyms()
    hyponyms = hyponyms[:4]
    for i in range(len(hyponyms)):
        candidate_distractor = " ".join(hyponyms[i].lemmas()[0].name().split('_'))
        res.append(f"{reserve} {candidate_distractor}")
    return res