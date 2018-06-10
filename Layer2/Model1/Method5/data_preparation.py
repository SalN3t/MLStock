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

import pprint
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

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import Word

from nltk.corpus import sentiwordnet as swn

# Load config
import json
config = json.load(open('../../../config.json'))

rx = re.compile('(["#\'\\%`])')
tb = Blobber(pos_tagger=NLTKTagger(), analyzer=NaiveBayesAnalyzer())

def add_to_dict(root_dicts, key, level):
    if root_dicts.has_key(key):
        root_dicts[ key ]  =  [root_dicts[ key ][0] + 1, level]
    else:
        root_dicts[ key ]  = [1,level]


def get_all_hypernyms(root_dicts,word):
    w = Word(word)
    reach_root = False
    for syns in w.synsets:
        reach_root = False
        i = 0
        while reach_root == False:
            try:
                if i == 0:
                    add_to_dict(root_dicts, syns.hypernyms()[0], i)
                elif i == 1:
                    add_to_dict(root_dicts, syns.hypernyms()[0].hypernyms()[0], i )
                elif i == 2:
                    add_to_dict(root_dicts, syns.hypernyms()[0].hypernyms()[0].hypernyms()[0], i  )
                elif i == 3:
                    add_to_dict(root_dicts, syns.hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0], i    )
                elif i == 4:
                    add_to_dict(root_dicts, syns.hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0] , i    )
                elif i == 5:
                    add_to_dict(root_dicts, syns.hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0] , i    )
                elif i == 6:
                    add_to_dict(root_dicts, syns.hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()[0] , i     )
                else:
                    reach_root = True
                i = i + 1
            except:
                    reach_root = True


## Process
stocks_symbols_file = config['stocks_symbols_file']
stocks_fullname_file = config['stocks_fullname_file']
twitter_dataset_directory = config['twitter_dataset_directory']
stocks_dataset_directory = config['stocks_dataset_directory']
stocks_dataset_suffex = config['stocks_dataset_suffex']
start_range_date = config['prepare']['range_filter_fot_stocks']['start_range_date']
end_range_date = config['prepare']['range_filter_fot_stocks']['end_range_date']
cleanup_output_dataset = config['prepare']['cleanup_output_dataset']
output_dataset_filename = config['prepare']['output_dataset_filename']
output_dataset_filename_arff = config['prepare']['output_dataset_filename_arff']
period = config['prepare']['period']

# ------------------------------------ Base Data Processing
# Because we only need the combined text and the stock status of wither it raise or fall we can use one of the datasets form the older methods 
# That should help not to redo the process again here
data_pd = pd.read_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method4/Method4_base_dataset.csv')

# Remove nan data if one exsists in the reduced text column from the dataset
data_pd = data_pd[ data_pd['reduced_text'].notnull() ]

# Make sure we don't have ## separator in text 
data_pd['reduced_text'].replace(' ## ','',inplace=True)

# ------------------------------------ Method 5 features build
# Load features choices and settings
features_config = json.load(open('features.json'))

all_words = features_config['words']
all_hypernms = features_config['hypernms']

if features_config['settings']['stem_words']:
    stemd = []
    for word in all_words:
        stemd.append( Word(word).stem().encode('utf8') )
    all_words = list(stemd)


# Convert list to dict with 0 value to be used as holder dict
words_holder_dict = dict((k,0) for k in all_words)
hypernms_holder_dict = dict((k,0) for k in all_hypernms)
#holder_dict.update(dict((k,0) for k in all_hypernms))

# features_list = []
# for key in holder_dict.keys():
#     features_list.append(key)


# Find the features in the txt and if it found make its value to be 1.. in here we will use binary 0: not found, 1: found
# Find the words features
output_list = []
for i,item in tqdm(data_pd['reduced_text'].iteritems()):
    item_features = dict(words_holder_dict)    # Make a copy of the dictianory with 0 values
    try:
        for word in item.strip().split(' '):
            word_1 = word.lower()
            if features_config['settings']['stem_words']:
                word_1 = Word(word).stem().lower()
            if word_1 in all_words:
                item_features[word_1] = 1
        #output_list.append([item,item_features, data_pd['stock_status'][i] ])
        item_dict = {'text':item,
            'stock_status' : data_pd['stock_status'][i],
            'stock_name': data_pd['stock_name'][i],
        }
        item_dict.update(item_features)
        output_list.append(item_dict)
    except:
        print("Data: %s" % item)
        break

# Find the hypernms features
for i,item in tqdm(data_pd['reduced_text'].iteritems()):
    item_features = dict(hypernms_holder_dict)    # Make a copy of the dictianory with 0 values
    for word in item.split(' '):
        word_1 = word.lower()
        tmp_hyper_dict = dict()
        get_all_hypernyms(tmp_hyper_dict, word_1)
        for feature in all_hypernms:
            for key_tmp in tmp_hyper_dict.keys():
                if feature.lower() in str(key_tmp).lower():
                    #print('Found: %s'% feature)
                    item_features[feature] = 1
    output_list[i].update(item_features)
    # output_list.append([item,item_features,item[1]])


# Find the useless features and remove them as we don't need those features that has 0 for all instance
# for item in output_list:
# Test feature use

# For words feature
new_tmp_dict = dict(words_holder_dict)
new_tmp_dict.update(hypernms_holder_dict)
for item in tqdm(output_list):
	for key_2 in item.keys():
		if item[key_2] == 1:
			new_tmp_dict[key_2] += 1

pprint.pprint(new_tmp_dict)

# Because this process takes lots of time (more than 2 hours) we will be saving the file so we can easily reget it if we need it
j_data = json.dumps(output_list)
with open("saved_checkpoint.json","w") as f:
  f.write(j_data)

# To load the data later we type
# with open("saved_checkpoint.json","r") as f:
#   d = json.load(f)

# Lets convert it to a format that we can then use to build dataframe

# for item in tqdm(output_list):
#     item[1].update({'text':item[0], 'stock_status':item[2]})

# final_list = list()
# for item in tqdm(output_list):
#     final_list.append( item[1] )

# Convert it to dataframe
final_pd = pd.DataFrame(output_list)

final_pd = final_pd[final_pd['stock_status'] != 'SAME']

# print data size
print('Data Size: ',len(final_pd))

final_pd_2 = final_pd.sort_values(by=['stock_name'])


final_pd_2.to_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method5/Method5_base_dataset.csv',index=False)



###################################################### ================= FIX ================= ##################################################


# # Find the features in the txt and if it found make its value to be 1.. in here we will use binary 0: not found, 1: found
# # Find the words features
# output_list = []
# for i,item in tqdm(data_pd['reduced_text'].iteritems()):
#     item_features = dict(words_holder_dict)    # Make a copy of the dictianory with 0 values
#     try:
#         for word in item.strip().split(' '):
#             word_1 = word.lower()
#             word_2 = word.lower()
#             if features_config['settings']['stem_words']:
#                 word_1 = Word(word).stem().lower()
#             if word_1 in all_words:
#                 item_features[word_1] = 1
#         output_list.append([item,item_features, data_pd['stock_status'][i] ])
#     except:
#         print("Data: %s" % item)
#         break

# # Find the hypernms features
# for i,item in tqdm(data_pd['reduced_text'].iteritems()):
#     item_features = dict(hypernms_holder_dict)    # Make a copy of the dictianory with 0 values
#     for word in item.split(' '):
#         word_1 = word.lower()
#         tmp_hyper_dict = dict()
#         get_all_hypernyms(tmp_hyper_dict, word_1)
#         for feature in all_hypernms:
#             for key_tmp in tmp_hyper_dict.keys():
#                 if feature.lower() in str(key_tmp).lower():
#                     #print('Found: %s'% feature)
#                     item_features[feature] = 1
#     output_list[i][1].update(item_features)
#     # output_list.append([item,item_features,item[1]])


# # Find the useless features and remove them as we don't need those features that has 0 for all instance
# # for item in output_list:
# # Test feature use

# # For words feature
# new_tmp_dict = dict(words_holder_dict)
# new_tmp_dict.update(hypernms_holder_dict)
# for item in tqdm(output_list):
# 	for key_2 in item[1].keys():
# 		if item[1][key_2] == 1:
# 			new_tmp_dict[key_2] += 1