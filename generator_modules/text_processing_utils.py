#this module contains functions required to process text corpus
import numpy as np 
import pandas as pd
import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import spacy
import zipfile
import os
import json
from sense2vec import Sense2Vec
import requests
from collections import OrderedDict
import string

import pke
import nltk
from nltk import FreqDist
nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from typing import List
import string


MIN_SENTENCE_LENGTH = 20

def tokenize_sentences(textComponent:str)-> List[str]:
    # get sentences from text
    sentences = sent_tokenize(textComponent)
    # remove sentences with length less than a minimum threshold
    sentences = [ sentence.strip() for sentence in sentences if len(sentence) > MIN_SENTENCE_LENGTH]
    return sentences

def  get_keywords_by_yake(textComponent:str,question_num:int=10):
    """extract relevant key words from text using YAKE key extraction algorithm"""
    results = []
    # create a YAKE extractor.
    extractor = pke.unsupervised.YAKE()
    # load the content of the text.
    stoplist = stopwords.get('english')
    extractor.load_document(input=textComponent,
                            language='en',
                            stoplist=stoplist,
                            normalization=None)
    # select {1-3}-grams not containing punctuation marks and not
    #    beginning/ending with a stopword as candidates.
    extractor.candidate_selection(n=3)
    # weight the candidates using YAKE weighting scheme, a window (in
    #    words) for computing left/right contexts can be specified.
    window = 2
    use_stems = False # use stems instead of words for weighting
    extractor.candidate_weighting(window=window,
                                    use_stems=use_stems)
    # get the 10-highest scored candidates as keyphrases.
    #    redundant keyphrases are removed from the output using levenshtein
    #    distance and a threshold.
    threshold = 0.8
    keyphrases = extractor.get_n_best(n=question_num, threshold=threshold)
    for key in keyphrases:
        results.append(key[0])
    return results

     
def get_keywords_by_multipartite(textComponent:str, question_num:int=10):
    """extract relevant key words from text using multipartite ranking"""
    results = []
    keyword_extractor = pke.unsupervised.MultipartiteRank()
    keyword_extractor.load_document(input=textComponent, language='en')
    stoplist = list(string.punctuation)
    stoplist += stopwords.words('english')
    keyword_extractor.candidate_selection(pos={'PROPN', 'NOUN'})
    try:
        keyword_extractor.candidate_weighting(alpha=1.1,threshold=0.74,method='average')
    except:
        return results

    keyphrases = keyword_extractor.get_n_best(n=question_num)

    for keyword in keyphrases:
        results.append(keyword[0])

    return results
        
def is_far(words_list,currentword,thresh,normalized_levenshtein):
    threshold = thresh
    score_list =[]
    for word in words_list:
        score_list.append(normalized_levenshtein.distance(word.lower(),currentword.lower()))
    if min(score_list)>=threshold:
        return True
    else:
        return False       
        
def filter_phrases(phrase_keys,max,normalized_levenshtein ):
    filtered_phrases =[]
    if len(phrase_keys)>0:
        filtered_phrases.append(phrase_keys[0])
        for ph in phrase_keys[1:]:
            if is_far(filtered_phrases,ph,0.7,normalized_levenshtein ):
                filtered_phrases.append(ph)
            if len(filtered_phrases)>=max:
                break
    return filtered_phrases

def MCQs_available(word,s2v):
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    if sense is not None:
        return True
    else:
        return False


def edits(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz '+string.punctuation
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def sense2vec_get_words(word,s2v):
    output = []

    word_preprocessed =  word.translate(word.maketrans("","", string.punctuation))
    word_preprocessed = word_preprocessed.lower()

    word_edits = edits(word_preprocessed)

    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=15)

    compare_list = [word_preprocessed]
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ")
        append_word = append_word.strip()
        append_word_processed = append_word.lower()
        append_word_processed = append_word_processed.translate(append_word_processed.maketrans("","", string.punctuation))
        if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
            output.append(append_word.title())
            compare_list.append(append_word_processed)


    out = list(OrderedDict.fromkeys(output))

    return out

def get_options(answer,s2v):
    distractors =[]

    try:
        distractors = sense2vec_get_words(answer,s2v)
        if len(distractors) > 0:
            print(" Sense2vec_distractors successful for word : ", answer)
            return distractors,"sense2vec"
    except:
        print (" Sense2vec_distractors failed for word : ",answer)


    return distractors,"None"


def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        word = word.strip()
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    delete_keys = []
    for k in keyword_sentences.keys():
        if len(keyword_sentences[k]) == 0:
            delete_keys.append(k)
    for del_key in delete_keys:
        del keyword_sentences[del_key]

    return keyword_sentences


def get_phrases(doc):
    phrases={}
    for np in doc.noun_chunks:
        phrase =np.text
        len_phrase = len(phrase.split())
        if len_phrase > 1:
            if phrase not in phrases:
                phrases[phrase]=1
            else:
                phrases[phrase]=phrases[phrase]+1

    phrase_keys=list(phrases.keys())
    phrase_keys = sorted(phrase_keys, key= lambda x: len(x),reverse=True)
    phrase_keys=phrase_keys[:50]
    return phrase_keys



def get_keywords(nlp,text,max_keywords,s2v,fdist,normalized_levenshtein,no_of_sentences):
    doc = nlp(text)
    max_keywords = int(max_keywords)

    keywords = get_keywords_by_multipartite(text=text)
    keywords = sorted(keywords, key=lambda x: fdist[x])
    keywords = filter_phrases(keywords, max_keywords,normalized_levenshtein )

    phrase_keys = get_phrases(doc)
    filtered_phrases = filter_phrases(phrase_keys, max_keywords,normalized_levenshtein )

    total_phrases = keywords + filtered_phrases

    total_phrases_filtered = filter_phrases(total_phrases, min(max_keywords, 2*no_of_sentences),normalized_levenshtein )


    answers = []
    for answer in total_phrases_filtered:
        if answer not in answers and MCQs_available(answer,s2v):
            answers.append(answer)

    answers = answers[:max_keywords]
    return answers
