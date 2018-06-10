import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import re
import string
import pickle
import os 
import datetime

import csv
import glob
from textblob import TextBlob
import nltk
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.taggers import NLTKTagger

# Setup
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('brown')
# nltk.download('movie_reviews')

# Load config
import json

rx = re.compile('(["#\'\\%`])')
tb = Blobber(pos_tagger=NLTKTagger(), analyzer=NaiveBayesAnalyzer())


data = pd.read_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer1_dataset/Model2/Layer1_base_dataset.csv')
# Keeping only the neccessary columns

data['headline'] = data['headline'].apply(lambda x: str(x).lower().replace(' ## ',''))

from nltk.corpus import sentiwordnet as swn
#result_reduce[1].split(',')[0]
def sent_from_text(text):
    test_b = tb( text )
    pos_count = 0.0
    neg_count = 0.0
    pos_sum = 0.0
    neg_sum = 0.0
    web_mention = 0.0
    for word in test_b.words:
        tmp_pos_score = 0.0
        tmp_neg_score = 0.0
        for item in swn.senti_synsets(word):
            tmp_pos_score = tmp_pos_score + item.pos_score()
            tmp_neg_score = tmp_neg_score +  item.neg_score()
            pos_sum = pos_sum + item.pos_score()
            neg_sum = neg_sum + item.neg_score()
        if 'http' in word:
            web_mention = web_mention + 1
        if str(tmp_pos_score) > str(tmp_neg_score):
            pos_count = pos_count + 1
        elif str(tmp_pos_score) < str(tmp_neg_score):
            neg_count = neg_count + 1
    return {'pos_count':int(round(pos_count)), 'neg_count': int(round(neg_count)) , 'pos_sum': int(round(pos_sum)) , 'neg_sum': int(round(neg_sum)), 'web_mention': int(round(web_mention)) }

from tqdm import tqdm
pos_sum = []
neg_sum = []

pos_count = []
neg_count = []

for item in tqdm(data['headline']):
    tmp_dict = sent_from_text(item)
    pos_sum.append(tmp_dict['pos_sum'])
    neg_sum.append(tmp_dict['neg_sum'])
    pos_count.append(tmp_dict['pos_count'])
    neg_count.append(tmp_dict['neg_count'])

data['pos_sum'] = pos_sum
data['neg_sum'] = neg_sum
data['pos_count'] = pos_count
data['neg_count'] = neg_count

data['sentiment_count_status'] = np.where( data['pos_count'] > data['neg_count'],'pos','neg'  )
data['sentiment_sum_status'] = np.where( data['pos_sum'] > data['neg_sum'],'pos','neg'  )

# 'P' if data['pos_count'] > data['neg_count'] else 'N'

file_name = '/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model2/Method1/Layer2_method1_base_dataset.csv'


data.to_csv(file_name, sep=',', encoding='utf-8',index=False)