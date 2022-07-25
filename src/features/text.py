#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/data/text.py

"""

from __future__ import absolute_import, division, print_function

import argparse, re, warnings
import numpy as np
import pandas as pd

from deep_translator import GoogleTranslator
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..utils.utils import * 

__author__ = "Fernando Pozo"
__copyright__ = "Copyright 2022"
__license__ = "GNU General Public License"
__version__ = "0.0.1"
__maintainer__ = "Fernando Pozo"
__email__ = "fpozoc@gmx.com"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser(
        description='Command-line arguments parser', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d", "--dataset",  
        help="Training set.")
    parser.add_argument(
        "-o", "--output",  
        help="Training set.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    df = pd.read_csv(args.dataset, lineterminator='\n')
    df['idx'] = range(1, len(df) + 1)
    df.insert(0, 'id', [f"{idx}.{totalRent} - {street} {houseNumber} {floor} ({town} - {city}, {state})" 
                for idx, state, city, town, street, houseNumber, floor, totalRent in zip(
                    df['idx'], df['regionLevel1'], df['regionLevel2'], df['regionLevel3'], df['street'], df['houseNumber'], df['floor'], df['totalRent'])])
    df = df.drop('idx', axis=1)

    df = nlp_feature_engineering(df=df, features=['description', 'facilities']).sample(60000)
    df = df[[
        'id', 'description', 'facilities', 
        'description_finbert', 'facilities_finbert', 
        'description_vader', 'facilities_vader', 
        'description_textblob_polarity', 'description_textblob_subjectivity', 
        'facilities_textblob_polarity', 'facilities_textblob_subjectivity']]
    df.to_csv(args.output, sep='\t', compression='gzip', index=False)


def nlp_feature_engineering(df:pd.DataFrame, features:list) -> pd.DataFrame:
    """NLP feature engineering.

    Args:
        df (pd.DataFrame): Dataframe with text to analyze.
        features (list): List of features to analyze.

    Returns:
        pd.DataFrame: Dataframe with analyzed features.
    """
    for feature in features:
        # translation
        df[feature] = batch_translator(list(df[feature].values))

        # Bert
        model = bert_model(model_path='yiyanghkust/finbert-tone', nlabels=3)
        df[f'{feature}_finbert'] = [huggingface_sentiment_analysis(pipeline=model, my_sentence=str(sentence)) for sentence in df[feature]]

        # VADER
        df[f'{feature}_vader'] = [vader_sentiment_analysis(my_sentence=str(sentence)) for sentence in df[feature]]

        # TextBlob
        df[f'{feature}_textblob'] = [textblob_sentiment_analysis(my_sentence=str(sentence)) for sentence in df[feature]]
        df[f'{feature}_textblob_polarity'], df[f'{feature}_textblob_subjectivity'] = df[f'{feature}_textblob'].str
        df.drop(f'{feature}_textblob', axis=1, inplace=True)
    return df


def batch_translator(text_list:list)->str:
    """Batch translate text from source language to target language.

    Args:
        text_list (list): List of text to translate.        
    
    Returns:
        translation_list(list): List of translated text.
    """
    text_list = list(map(lambda x: str(x).replace('nan', ''), [re.sub('[^a-zA-Z0-9 \.]', '', str(my_sentence)) for my_sentence in text_list]))
    return GoogleTranslator('de', 'en').translate_batch(text_list)


def bert_model(model_path:str, nlabels:int=3)->object:
    """Bert model.

    Args:
        model_path (str): Bert model path.
        nlabels (int): Number of labels.

    Returns:
        object: Bert model.
    """
    finbert = BertForSequenceClassification.from_pretrained(model_path,num_labels=nlabels)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return pipeline("text-classification", model=finbert, tokenizer=tokenizer)


def huggingface_sentiment_analysis(pipeline:object, my_sentence:str)->tuple:
    """HuggingFace NLP pipeline.

    Args:
        pipeline (object): HuggingFace NLP pipeline.
        my_sentence (str): Text to analyze.

    Returns:
        tuple
    """
    if len(my_sentence) > 3: # avoid nan's and annotations
        nlp_dict = pipeline(my_sentence)[0]
        label = nlp_dict['label']
        score = nlp_dict['score']
        if label == 'Neutral':
            result = 0
        elif label == 'Negative':
            result = score*-1
        elif label == 'Positive':
            result = score
        else:
            result = 0
    else:
        # result = 0
        label = 0
    return label


def textblob_sentiment_analysis(my_sentence:str)->tuple:
    """Textblob sentiment analysis.

    Args:
        my_sentence (str): Text to analyze.
    
    Returns:
        tuple: (polarity, subjectivity)
    """
    if len(my_sentence)>3:  # avoid nan's and annotations
        blob = TextBlob(my_sentence)
        polarity = blob.sentiment.polarity    
        subjectivity = blob.sentiment.subjectivity
        result = (polarity, subjectivity)
    else:
        result = (0,0)
    return result


def vader_sentiment_analysis(my_sentence:str)->tuple:
    """Vader sentiment analysis.

    Args:
        my_sentence (str): Text to analyze.

    Returns:
        result (float): polarity and subjectivity summary.
    """
    if len(my_sentence)>3:  # avoid nan's and annotations
        result = SentimentIntensityAnalyzer().polarity_scores(my_sentence)['compound']
    else:
        result = 0
    return result

    
if __name__ == "__main__":
    main()