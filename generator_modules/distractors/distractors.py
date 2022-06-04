from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sense2vec import Sense2Vec
from generator_modules.utils import QuestionType, ErrorMessages
from generator_modules.models import Question, DistractorRequest
from generator_modules.text_processing_utils import get_options


class DistractorGenerator:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(self.device)
        self.s2v = Sense2Vec().from_disk('s2v_old')
        
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
                options = get_options(answer, self.s2v)
                question_object = Question(question=question, answer=answer, options=options, question_object=QuestionType.MCQ)
            else:
                question_object = Question(question=question, answer="Not found", options=[], question_object=QuestionType.MCQ)
            result.append(question_object.dict())
            
        return result
            