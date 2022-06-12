from transformers import pipeline


class Summarizer:
    def __init__(self):
        self.summarizer_model = pipeline("summarization")
    
    def summarize(self,text, min_length=5, max_length=50):
        try:
            summary = self.summarizer_model(text,max_length=max_length, min_length=min_length, do_sample=False)
        except:
            summary = [{"summary_text":""}]
        return summary