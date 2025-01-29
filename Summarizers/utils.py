import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
import os
import numpy as np

def calculate_mean(df, column_name='highlights'):
    """
    Calculate the mean number of sentences across all summaries.
    """
    df[f'n_sentences{column_name}'] = df[column_name].apply(lambda x: len(sent_tokenize(x)))
    mean_sentences = df[f'n_sentences{column_name}'].mean()
    
    return df, mean_sentences


# Preprocess data
def cleaned_list_of_sentences(article):
    # Split text into sentences
    sentences = sent_tokenize(article)

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    filtered_sentences = []
    for sent in sentences:
        filtered_stemmed_tokens = []
        tokens = word_tokenize(sent)
        for word in tokens:
            word = stemmer.stem(word) 
            if word not in stop_words:
                filtered_stemmed_tokens.append(word)

        filtered_sentence = " ".join(filtered_stemmed_tokens)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


def calculate_scores(df, summary_col, reference_col):    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

    all_scores = []

    for idx, row in df.iterrows():
        candidate = row[summary_col]
        reference = row[reference_col]
        
        scores = scorer.score(candidate, reference) # compute rough
        
        all_scores.append(scores)

    df['rouge_scores'] = all_scores

    return df

# def sum_metrices(df, metrics_column='rouge_scores', results_folder='Results', file_name='metrics_results.csv'):
#     if not os.path.exists(results_folder):
#         os.makedirs(results_folder)

#     metric_data = {
#         'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
#         'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
#         'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
#         'rougeLsum': {'precision': [], 'recall': [], 'fmeasure': []},
#     }

#     # Individual metric values
#     for scores in df[metrics_column]:
#         for rouge_type, metric in scores.items():
#             metric_data[rouge_type]['precision'].append(metric.precision)
#             metric_data[rouge_type]['recall'].append(metric.recall)
#             metric_data[rouge_type]['fmeasure'].append(metric.fmeasure)

#     # Calculate mean and std
#     metrics_summary = {rouge_type: {key: {'mean': np.mean(values), 'std': np.std(values, ddof=1)}
#             for key, values in metrics.items()}
#             for rouge_type, metrics in metric_data.items()
#     }

#     results_data = []
#     for rouge_type, metrics in metrics_summary.items():
#         for key, stats in metrics.items():
#             results_data.append({
#                 'Metric': rouge_type,
#                 'Type': key,
#                 'Mean': stats['mean'],
#                 'StdDev': stats['std']
#             })

#     results_df = pd.DataFrame(results_data)
#     results_file = os.path.join(results_folder, file_name)
#     results_df.to_csv(results_file, index=False)

#     return metrics_summary
def sum_metrices(df, metrics_column='rouge_scores', results_folder='Results', file_name='metrics_results.csv'):
    """
    Summarize and save only overall ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum).
    """
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Store mean F1 scores for each ROUGE type
    metric_data = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': [],
    }

    # Collect F1 scores for each ROUGE type
    for scores in df[metrics_column]:
        for rouge_type, metric in scores.items():
            metric_data[rouge_type].append(metric.fmeasure)

    # Calculate mean and std for F1 scores
    metrics_summary = {
        rouge_type: {
            'mean_f1': np.mean(values),
            'std_f1': np.std(values, ddof=1)
        }
        for rouge_type, values in metric_data.items()
    }

    # Prepare results for saving
    results_data = [
        {
            'Metric': rouge_type,
            'Mean_F1': stats['mean_f1'],
            'StdDev_F1': stats['std_f1']
        }
        for rouge_type, stats in metrics_summary.items()
    ]

    # Save to CSV
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(results_folder, file_name)
    results_df.to_csv(results_file, index=False)

    return metrics_summary


def calculate_number_of_seneteces(df, column_name='highlights'):
    """
    Calculate the number of sentences in column and the mean number of sentences across all summaries.
    """
    df[f'n_sentences{column_name}'] = df[column_name].apply(lambda x: len(sent_tokenize(x)))
    mean_sentences = df[f'n_sentences{column_name}'].mean()
    
    return df, mean_sentences

