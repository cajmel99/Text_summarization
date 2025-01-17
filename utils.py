from rouge import Rouge
from datasets import load_dataset
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import regex as re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer

# Preprocess data
def cleaned_list_of_sentences(article):
    # Split text into sentences
    sentences = sent_tokenize(article)

    # Prepare the set of stop words
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # For each sentence, tokenize -> remove stopwords -> rejoin
    filtered_sentences = []
    for sent in sentences:
        filtered_stemmed_tokens = []
        # Tokenize words in this sentence
        tokens = word_tokenize(sent)
        # Apply steaming and filter out stopwords 
        for word in tokens:
            word = stemmer.stem(word) 
            if word not in stop_words:
                filtered_stemmed_tokens.append(word)

        # Rejoin tokens to form a sentence
        filtered_sentence = " ".join(filtered_stemmed_tokens)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences
