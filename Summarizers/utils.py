from rouge import Rouge
from datasets import load_dataset
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
import os

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


def calculate_scores(df, summary_col, reference_col):    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

    all_scores = []

    for idx, row in df.iterrows():
        candidate = row[summary_col]
        reference = row[reference_col]
        
        # Compute the ROUGE scores for this pair
        scores = scorer.score(candidate, reference)
        
        # Each score is a dict of { 'precision': float, 'recall': float, 'fmeasure': float }
        all_scores.append(scores)

    # Attach these raw scores back to the DataFrame as a new column
    df['rouge_scores'] = all_scores

    return df

def sum_metrices(df, metrics_column='rouge_scores', results_folder='Results', file_name='metrics_results.csv'):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Dictionary for metrics sums
    metric_sums = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeLsum': {'precision': 0, 'recall': 0, 'fmeasure': 0},
    }
    
    metrics_count = {key: 0 for key in metric_sums}

    # Sum up the metrics
    for scores in df[metrics_column]:
        for rouge_type, metric in scores.items():
            metric_sums[rouge_type]['precision'] += metric.precision
            metric_sums[rouge_type]['recall'] += metric.recall
            metric_sums[rouge_type]['fmeasure'] += metric.fmeasure
            metrics_count[rouge_type] += 1

    # Calculate the mean of each metric
    metrics_mean = {
        rouge_type: {key: value / metrics_count[rouge_type] for key, value in metric.items()}
        for rouge_type, metric in metric_sums.items()
    }

    results_file = os.path.join(results_folder, file_name)    
    metrics_df = pd.DataFrame(metrics_mean).transpose()
    metrics_df.to_csv(results_file, index=True)
    
    return metrics_mean
