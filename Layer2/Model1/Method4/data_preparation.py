try:
    from utility import *
except:
    import os,sys
    sys.path.insert(0,os.path.dirname( os.path.dirname( os.path.dirname(os.getcwd() ) ) ) )
    from utility import *

import re
import string
import pickle
import os 

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import csv
import glob
from textblob import TextBlob
import nltk
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.taggers import NLTKTagger

from nltk.corpus import sentiwordnet as swn

# Load config
import json
config = json.load(open('../../../config.json'))

# rx = re.compile('(["#\'\\%`])')

data_pd = pd.read_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method1/Method1_base_dataset.csv')

count_dict = dict()
for item in data_pd['stock_name']:
    if count_dict.has_key(item):
        count_dict[item] += 1
    else:
        count_dict[item] = 1

# find all the stocks that has less than 100 instance and remove them
remove_list = list()
for i,item in data_pd['stock_name'].iteritems():
    if count_dict[item] < 100:
        remove_list.append(i)

print('removing: ',len(remove_list))

data_pd_2 = data_pd.drop(data_pd.index[ remove_list ])

# Write the dataset
data_pd_2.to_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method4/Method4_base_dataset.csv',index=False)
