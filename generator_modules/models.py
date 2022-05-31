from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    text:str
    num_question: int =5
    optional_keywords: List[str] =[]
    
    
class Question(BaseModel):
    question:str
    answer: str
    options: List[str]
    question_type:str

