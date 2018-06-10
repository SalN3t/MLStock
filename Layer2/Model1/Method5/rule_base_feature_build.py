try:
    from utility import *
except:
    import os,sys
    sys.path.insert(0,os.path.dirname( os.path.dirname( os.path.dirname(os.getcwd() ) ) ) )
    from utility import *


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import operator

from tqdm import tqdm
from subprocess import check_output
from nltk.tokenize import word_tokenize
from textblob import Word
stopwords_set = set(stopwords.words("english"))


def from_text_to_words_list(r_data, term):
    words_list = list()
    for item in r_data:
        if term in item[1]:
            words = item[0].split(' ')
            for w in words:
                if not w in stopwords_set:
                    words_list.append(w)
    return words_list

def from_text_to_words_list_panda(data_pd, term, text_index = 'reduced_text', status_index = 'stock_status'):
    words_list = list()
    for item in data_pd[data_pd[status_index] == term ][text_index]:
        words = item.split(' ')
        for w in words:
            if not w in stopwords_set:
                words_list.append(w)
    return words_list

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
 
# def get_words_in_tweets(tweets):
#     all = []
#     for (words, sentiment) in tweets:
#         all.extend(words)
#     return all

def get_word_features(wordlist, top_level = -1):
    wordlist = nltk.FreqDist(wordlist)
    features = sorted(wordlist.items(), key=operator.itemgetter(1), reverse=True)[:top_level] # will return tuple (word, frequency)
    return [ feature[0] for feature in features]

# def extract_features(document):
#     document_words = set(document)
#     features = {}
#     for word in w_features:
#         features['containts(%s)' % word] = (word in document_words)
#     return features

def wordcloud_draw(data, color = 'black', image_name = 'Figure.jpg'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    fig1 = plt.gcf()
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    fig1.savefig(image_name)
    # fig1.savefig(image_name,dpi=100)


def add_to_dict(root_dicts, key, level):
    if root_dicts.has_key(key):
        root_dicts[ key ]  =  [root_dicts[ key ][0] + 1, level]
    else:
        root_dicts[ key ]  = [1,level]


def get_all_hypernyms(root_dicts,word):
    w = Word(word)
    #root_dicts = dict()
    output_list = list()
    reach_root = False
    for syns in w.synsets:
        reach_root = False
        i = 0
        while reach_root == False:
            try:
                if i == 0:
		            #output_list.append(syns.hypernyms()[0])
                    add_to_dict(root_dicts, syns.hypernyms()[0], i)
                elif i == 1:
		            #output_list.append(syns.hypernyms()[0].hypernyms()[0])
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


# data = handle_file('/home/salah/school/Capston/gitlab_data/MLsTock/datasets/finished_dataset/Dataset_final_stock_5_7_18_mod_w_sent_reduced.csv','r')
# r_data = [( item.split(',')[0],item.split(',')[1])  for item in data]

# ------------------------------------ Base Data Processing
# Because we only need the combined text and the stock status of wither it raise or fall we can use one of the datasets form the older methods 
# That should help not to redo the process again here
data_pd = pd.read_csv('/media/salah/e58c5812-2860-4033-90c6-83b7ffaa8b88/MLStock/dataset/Layer2_dataset/Model1/Method4/Method4_base_dataset.csv')

# Remove nan data if one exsists in the reduced text column from the dataset
data_pd = data_pd[ data_pd['reduced_text'].notnull() ]

# ------------------------------------ Feature Extraction
# Find Words Feature to pick from
train_pos = from_text_to_words_list_panda(data_pd, 'RAISE')
train_neg = from_text_to_words_list_panda(data_pd, 'FALL')
in_words = intersection(train_pos, train_neg)

print('Pos Words len: %d, Neg Wrods len: %d, intersection len: %d'% (len(train_pos), len(train_neg), len(in_words)))

wordcloud_draw ( get_word_features(train_pos, top_level = 20000), 'white' , image_name='random_output_data/Figure1_pos_words.jpg')
wordcloud_draw ( get_word_features(train_neg, top_level = 20000) ,image_name='random_output_data/Figure2_neg_words.jpg')

# Remove all the words that both sets shares and keep only the unique ones for that set
train_pos = [word for word in tqdm(train_pos) if not word in in_words]
train_neg = [word for word in tqdm(train_neg) if not word in in_words]

print('After reduce:\nPos Words len: %d, Neg Wrods len: %d'% (len(train_pos), len(train_neg) ))

wordcloud_draw ( get_word_features(train_pos, top_level = 20000), 'white' , image_name='random_output_data/Figure3_pos_words_after_intersection_reduce.jpg')
wordcloud_draw ( get_word_features(train_neg, top_level = 20000) ,image_name='random_output_data/Figure4_neg_words_after_intersection_reduce.jpg')

pos_freq_words_dict = nltk.FreqDist(train_pos)
neg_freq_words_dict = nltk.FreqDist(train_neg)

# Convert it to list in order of the highest frequency first
pos_freq_words_list = sorted(pos_freq_words_dict.iteritems(), key=lambda (k,v): (v,k), reverse=True)
neg_freq_words_list = sorted(neg_freq_words_dict.iteritems(), key=lambda (k,v): (v,k), reverse=True)

# Write to file
write_file(pos_freq_words_list, filename = 'random_output_data/pos_words_frequency_features.txt')
write_file(pos_freq_words_list, filename = 'random_output_data/neg_words_frequency_features.txt')

# w_features = get_word_features(get_words_in_tweets(train), top_level = 20000 )

# Find Hypernms features to pick from
pos_hypernms_dict = dict()
for word in tqdm(train_pos):
    get_all_hypernyms(pos_hypernms_dict,word)

neg_hypernms_dict = dict()
for word in tqdm(train_pos):
    get_all_hypernyms(neg_hypernms_dict,word)

# Convert it to list in order of the highest frequency first
pos_hypernms_sorted_list = sorted(pos_hypernms_dict.iteritems(), key=lambda (k,v): (v[1],v[0]), reverse=True)
neg_hypernms_sorted_list = sorted(neg_hypernms_dict.iteritems(), key=lambda (k,v): (v[1],v[0]), reverse=True)

# Write to file
write_file(pos_hypernms_sorted_list, filename = 'random_output_data/pos_hypernms_frequency_features.txt')
write_file(neg_hypernms_sorted_list, filename = 'random_output_data/neg_hypernms_frequency_features.txt')

# ===========================
# # Find Hypernms features to pick from

# # Find Words Feature to pick from
# train_pos = from_text_to_words_list(r_data, 'RAISE')
# train_neg = from_text_to_words_list(r_data, 'FALL')
# in_words = intersection(train_pos, train_neg)

# # Remove all the words that both sets shares and keep only the unique ones for that set
# train_pos = [word for word in train_pos if not word in in_words]
# train_neg = [word for word in train_neg if not word in in_words]

# wordcloud_draw ( get_word_features(train_pos, top_level = 20000), 'white' )
# wordcloud_draw ( get_word_features(train_neg, top_level = 20000) )

# # w_features = get_word_features(get_words_in_tweets(train), top_level = 20000 )

# # Find Hypernms features to pick from
